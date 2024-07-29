import torch, time
from typing import Optional
import numpy as np
from . import comm
from . import profiler

def gpt_speculative_decoding(
        gpt, 
        draft_gpt, 
        draft_steps_max: int,
        vocab_size: int,
        input_token_ids: torch.IntTensor,
        input_lengths: torch.IntTensor,
        gen_length: int,
        eos_token_id: Optional[int] = None,
        local_batch_size: Optional[int] = None,
        beam_width: int = 1,
        top_k: Optional[torch.IntTensor] = None,
        top_p: Optional[torch.FloatTensor] = None,
        top_p_decay: Optional[torch.FloatTensor] = None,
        top_p_min: Optional[torch.FloatTensor] = None,
        top_p_reset_ids: Optional[torch.IntTensor] = None,
        temperature: Optional[torch.FloatTensor] = None,
        repetition_penalty: Optional[torch.FloatTensor] = None,
        presence_penalty: Optional[torch.FloatTensor] = None,
        min_length: Optional[torch.IntTensor] = None,
        len_penalty: Optional[torch.FloatTensor] = None,
        beam_search_diversity_rate: Optional[torch.FloatTensor] = None,
        stop_words_list: Optional[torch.IntTensor] = None,
        bad_words_list: Optional[torch.IntTensor] = None,
        sequence_limit_lengths: Optional[torch.IntTensor] = None,
        random_seed: Optional[torch.LongTensor] = None,
        memory_length: Optional[int] = None,
        return_output_length: bool = False,
        return_log_probs: bool = False):
    assert beam_width == 1, "only support beam_width = 1"
    assert gpt.weight is not None and draft_gpt.weight is not None, 'Please call load() first to initialize weights.'
    
    input_token_ids = input_token_ids.type(torch.int32).to(gpt.device)
    input_lengths = input_lengths.type(torch.int32).to(gpt.device)

    batch_size = len(input_token_ids)
    max_input_length = input_token_ids.shape[-1]
    max_seq_length = max_input_length + gen_length + draft_steps_max
    memory_length = memory_length or max_seq_length

    assert local_batch_size is None or local_batch_size == batch_size
    local_batch_size = batch_size
    num_local_batches = 1

    device = gpt.device

    eos_token_id = eos_token_id if eos_token_id is not None else gpt.config.end_id
    assert eos_token_id is not None, 'eos_token-id must be specified in generation.'
    eos_token_ids = eos_token_id * torch.ones(batch_size, dtype=torch.int32, device=device)
    assert repetition_penalty is None or presence_penalty is None,\
        'Found ambiguous parameters repetition_penalty and presence_penalty '\
        'which are mutually exclusive. Please provide one of repetition_penalty '\
        'and presence_penalty.'

    # Prepare input and output arguments.
    if beam_width > 1:
        # Tiling for beam search.
        input_token_ids = input_token_ids.repeat(1, beam_width).view(batch_size * beam_width, -1)
        input_lengths = input_lengths.view(-1, 1).repeat(1, beam_width).view(-1)
        if sequence_limit_lengths is not None:
            sequence_limit_lengths = sequence_limit_lengths.view(-1, 1).repeat(1, beam_width).view(-1)
        # src/tgt cache indirections.
        cache_indirection = torch.zeros(
            (2, batch_size, beam_width, memory_length), dtype=torch.int32, device=device)
        parent_ids = torch.zeros(max_seq_length, batch_size * beam_width, dtype=torch.int32, device=device)
    else:
        cache_indirection = None
        src_cache_indirection = None
        tgt_cache_indirection = None
        parent_ids = None

    pad_lengths = max_input_length - input_lengths
    # Since tril() doesn't support bf16 dtype, we create of bool type and then cast it to dtype.
    attention_mask = torch.ones(
        (max_input_length, max_input_length), dtype=torch.bool, device=device)\
        .tril().unsqueeze(0).tile(input_token_ids.shape[0], 1, 1).to(gpt.dtype)
    for b, input_length in enumerate(input_lengths):
        attention_mask[b, input_length:, ...] = 0
    masked_tokens = gpt.generate_pad_mask(input_lengths, memory_length)
    finished = torch.zeros_like(input_lengths).bool()
    sequence_lengths = input_lengths.clone()-1 # sequence_lengths 是最后一个有效token的位置
    if return_log_probs or beam_width > 1:
        cum_log_probs = torch.zeros(batch_size * beam_width, device=device)
        output_log_probs = torch.zeros((gen_length, batch_size * beam_width), device=device)
    else:
        cum_log_probs = None
        output_log_probs = None

    # Contiguous buffer for each decode_op step, it will be transposed tensor for the final output.
    global_position_ids = torch.clamp(torch.arange(max_seq_length+draft_steps_max*2, dtype=torch.int32, device=device).unsqueeze(1).repeat(1,batch_size), max=2046)
    output_token_ids = torch.zeros(
        (max_seq_length+draft_steps_max*2, batch_size * beam_width), dtype=torch.int32, device=device)
    output_token_ids[:max_input_length, ...] = input_token_ids.T

    torch.cuda.synchronize()
    start_time = time.time()*1000
    if comm.is_pipeline_group_first():
        # Prepare input tensors of decoder.
        input_embeds = gpt.word_embedding(input_token_ids)
        if gpt.position_encoding is not None:
            position_ids = torch.arange(0, max_input_length, dtype=torch.int, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, max_input_length)
            input_embeds += gpt.position_encoding(position_ids)
        if gpt.pre_decoder_layernorm is not None:
            input_embeds = gpt.pre_decoder_layernorm(input_embeds)

        draft_input_embeds = draft_gpt.word_embedding(input_token_ids)
        if draft_gpt.position_encoding is not None:
            position_ids = torch.arange(0, max_input_length, dtype=torch.int, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, max_input_length)
            draft_input_embeds += draft_gpt.position_encoding(position_ids)
        if draft_gpt.pre_decoder_layernorm is not None:
            draft_input_embeds = draft_gpt.pre_decoder_layernorm(draft_input_embeds)
    else:
        # Dummy input_embeds
        input_embeds = torch.empty(
            size=(batch_size * beam_width, max_input_length, gpt.context_decoder.hidden_size),
            dtype=gpt.context_decoder.dtype,
            device=device)
        draft_input_embeds = torch.empty(
            size=(batch_size * beam_width, max_input_length, gpt.context_decoder.hidden_size),
            dtype=draft_gpt.context_decoder.dtype,
            device=device)

    use_shared_contexts = (gpt.shared_contexts_ratio > 0.) and (max_input_length >= 1) and (batch_size > 1)
    batch_to_compact, compact_to_batch = None, None
    if use_shared_contexts:
        find_context_duplications = torch.ops.fastertransformer.find_context_duplications
        batch_to_compact, compact_to_batch = find_context_duplications(input_token_ids)
        use_shared_contexts = compact_to_batch.shape[0] <= gpt.shared_contexts_ratio * batch_size

        if not use_shared_contexts:
            batch_to_compact, compact_to_batch = None , None

    profiler.start('ft-context-decoder')
    _, k_cache, v_cache, last_token_hidden_states = gpt.context_decoder.forward(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        memory_length=memory_length,
        batch_to_compact_index=batch_to_compact,
        compact_index=compact_to_batch)
    _, draft_k_cache, draft_v_cache, draft_last_token_hidden_states = draft_gpt.context_decoder.forward(
        input_embeds=draft_input_embeds,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        memory_length=memory_length,
        batch_to_compact_index=batch_to_compact,
        compact_index=compact_to_batch)
    profiler.stop('ft-context-decoder')

    torch.cuda.synchronize()
    context_time = time.time()*1000
    wall_time_list = [context_time-start_time]
    profiler.start('ft-decoder-all')
    masked_tokens[:] = False

    # for step in range(max_input_length, max_seq_length):
    first_context_decoding = True
    inference_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
    token_nums_per_sample = torch.ones((batch_size), dtype=torch.int32, device=input_embeds.device)
    all_accept = torch.zeros(batch_size, dtype=torch.int32, device=device)
    for step in range(max_input_length, max_seq_length):
        inference_steps += (1-finished.int())
        finished_np = finished.cpu().numpy()
        for ite in range(num_local_batches):
            draft_start_time, draft_end_time, decoder_end_time, post_process_end_time = 0,0,0,0
            # # The indices of the current local batch-beam.
            bbidx = np.arange(
                ite * local_batch_size * beam_width,
                min((ite + 1) * local_batch_size * beam_width, batch_size * beam_width))
            
            torch.cuda.synchronize()
            draft_start_time = time.time()*1000
            if first_context_decoding:
                hidden_states = last_token_hidden_states[bbidx, ...]
            else:
                if comm.is_pipeline_group_first():
                    # draft model predict
                    draft_sequence_lengths = sequence_lengths.clone()
                    draft_sequence_lengths_cpu = draft_sequence_lengths.cpu().numpy()
                    profiler.start('ft-decoder-draft')
                    
                    for draft_step in range(draft_steps_max):
                        if draft_step ==0 and all_accept.sum()>0:
                            profiler.start('ft-decoder-draft1')
                            input_tokens = torch.stack([output_token_ids[draft_sequence_lengths_cpu-1,bbidx], output_token_ids[draft_sequence_lengths_cpu,bbidx]]).T.flatten()
                            position_ids = torch.stack([global_position_ids[draft_sequence_lengths_cpu-1,bbidx], global_position_ids[draft_sequence_lengths_cpu,bbidx]]).T.flatten()
                            token_nums_per_sample[:] = 2
                            profiler.stop('ft-decoder-draft1')
                        else:
                            profiler.start('ft-decoder-draft2')
                            input_tokens = output_token_ids[draft_sequence_lengths_cpu, bbidx]
                            position_ids = global_position_ids[draft_sequence_lengths_cpu, bbidx]
                            token_nums_per_sample[:] = 1
                            profiler.stop('ft-decoder-draft2')
                        input_embeds = draft_gpt.word_embedding(input_tokens)

                        if draft_gpt.position_encoding is not None:
                            input_embeds += draft_gpt.position_encoding(position_ids)
                        if draft_gpt.pre_decoder_layernorm is not None:
                            input_embeds = draft_gpt.pre_decoder_layernorm(input_embeds)
                        hidden_states = draft_gpt.decoder.forward(
                            max_input_length=max_input_length,
                            step=0,
                            ite=ite,
                            input_embeds=input_embeds,
                            sequence_lengths=sequence_lengths[bbidx]-1 if (draft_step ==0 and all_accept.sum()>0) else sequence_lengths[bbidx]+draft_step,
                            key_cache=draft_k_cache,
                            value_cache=draft_v_cache,
                            finished=finished[bbidx],
                            total_padding_tokens=pad_lengths[bbidx],
                            cache_indirection=src_cache_indirection,
                            masked_tokens=masked_tokens[bbidx, ...],
                            token_nums_per_sample=token_nums_per_sample)

                        if draft_step ==0 and all_accept.sum()>0:
                            hidden_states = hidden_states.reshape(batch_size, 2, -1)
                            hidden_states = hidden_states[:,-1,:]
                        if draft_gpt.post_decoder_layernorm is not None:
                            hidden_states = draft_gpt.post_decoder_layernorm(hidden_states)

                        # We use logits of fp32 type to avoid overflow issue.
                        if draft_gpt.use_fp32_to_compute_logit:
                            # The FT GPT op internally uses FP32 compute type for matrix multiplication.
                            # This will produce the same result with the end-to-end FT's GPT op.
                            logits = torch.nn.functional.linear(hidden_states.float(), draft_gpt.lm_head.weight)[:,:vocab_size]
                        else:
                            logits = draft_gpt.lm_head(hidden_states).float()[:,:vocab_size]
                        predict_tokens = logits.argmax(-1).type(torch.int32)
                        output_token_ids[draft_sequence_lengths_cpu+1, bbidx] = predict_tokens
                        draft_sequence_lengths_cpu += 1

                    profiler.stop('ft-decoder-draft')
                    
                    profiler.start('ft-decoder-prepare')
                    draft_sequence_lengths_cpu += 1
 
                    input_tokens_list, position_ids_list = [], []
                    for idx in range(batch_size):
                        if finished[idx]:
                            token_nums_per_sample[idx] = 0
                        else:
                            s,e = sequence_lengths[idx].item(),draft_sequence_lengths_cpu[idx]
                            input_tokens_list.append(output_token_ids[s:e, idx])
                            position_ids_list.append(global_position_ids[s:e, idx])
                            token_nums_per_sample[idx] = e-s
                    input_tokens = torch.cat(input_tokens_list, dim=0)
                    position_ids = torch.cat(position_ids_list, dim=0)
                    input_embeds = gpt.word_embedding(input_tokens)
                    if gpt.position_encoding is not None:
                        input_embeds += gpt.position_encoding(position_ids)
                    if gpt.pre_decoder_layernorm is not None:
                        input_embeds = gpt.pre_decoder_layernorm(input_embeds)
                    profiler.stop('ft-decoder-prepare')
                else:
                    # Dummy input_imbeds
                    input_embeds = torch.empty(
                        size=(len(bbidx), gpt.decoder.hidden_size),
                        dtype=gpt.decoder.dtype,
                        device=device)
                torch.cuda.synchronize()
                draft_end_time = time.time()*1000
                profiler.start('ft-decoder')
                hidden_states = gpt.decoder.forward(
                    max_input_length=max_input_length,
                    step=0,
                    ite=ite,
                    input_embeds=input_embeds,
                    sequence_lengths=sequence_lengths[bbidx],
                    key_cache=k_cache,
                    value_cache=v_cache,
                    finished=finished[bbidx],
                    total_padding_tokens=pad_lengths[bbidx],
                    cache_indirection=src_cache_indirection,
                    masked_tokens=masked_tokens[bbidx, ...],
                    token_nums_per_sample=token_nums_per_sample)
                profiler.stop('ft-decoder')
            torch.cuda.synchronize()
            decoder_end_time = time.time()*1000
            if comm.is_pipeline_group_last():
                if gpt.post_decoder_layernorm is not None:
                    hidden_states = gpt.post_decoder_layernorm(hidden_states)

                # We use logits of fp32 type to avoid overflow issue.
                if gpt.use_fp32_to_compute_logit:
                    # The FT GPT op internally uses FP32 compute type for matrix multiplication.
                    # This will produce the same result with the end-to-end FT's GPT op.
                    logits = torch.nn.functional.linear(hidden_states.float(), gpt.lm_head.weight)
                else:
                    logits = gpt.lm_head(hidden_states).float()
                
                profiler.start('ft-decoder-post-process')
                predict_tokens = logits[:,:vocab_size].argmax(-1).type(torch.int32)
                if first_context_decoding:
                    output_token_ids[sequence_lengths.cpu().numpy()+1, bbidx] = predict_tokens.type(torch.int32)
                    sequence_lengths += 1

                    finished = finished | (predict_tokens==eos_token_id)

                    first_context_decoding = False
                else:
                    # verify
                    batch_offset = 0
                    for idx, token_nums in enumerate(token_nums_per_sample):
                        token_nums_item = token_nums.item()
                        if token_nums_item == 0:
                            continue
                        input_tokens_idx = input_tokens[batch_offset:batch_offset+token_nums_item]
                        predict_tokens_idx = predict_tokens[batch_offset:batch_offset+token_nums_item]
                        accept_length = 1
                        for i in range(token_nums_item):
                            if i==token_nums_item-1 or predict_tokens_idx[i] != input_tokens_idx[i+1] or predict_tokens_idx[i]==eos_token_id:
                                accept_length = 1+i
                                break

                        sequence_lengths[idx] += accept_length
                        output_token_ids[sequence_lengths[idx].item(), idx] = predict_tokens_idx[accept_length-1]
                        if predict_tokens_idx[accept_length-1] == eos_token_id or sequence_lengths[idx]+1-input_lengths[idx]>=gen_length:
                            finished[idx] = True
                        batch_offset += token_nums_item
                        if accept_length == token_nums_item and not finished[idx]:
                            all_accept[idx] = 1
                        else:
                            all_accept[idx] = 0
                profiler.stop('ft-decoder-post-process')
                torch.cuda.synchronize()
                post_process_end_time = time.time()*1000
                if step == max_input_length:
                    wall_time_list.extend([decoder_end_time-draft_start_time, post_process_end_time-decoder_end_time])
                else:
                    wall_time_list.extend([draft_end_time-draft_start_time, decoder_end_time-draft_end_time, post_process_end_time-decoder_end_time])
        if finished.all():
            break
    profiler.stop('ft-decoder-all')
    # Transpose (L, batch, beam) -> (batch, beam, L)
    output_token_ids = output_token_ids.view(-1, batch_size, beam_width).permute(1, 2, 0)

    torch.cuda.synchronize()
    end_time = time.time()*1000

    # Outputs
    output_dict = dict(output_token_ids=output_token_ids)
    output_dict['inference_steps'] = inference_steps
    output_dict['wall_time'] = wall_time_list
    output_dict['pad_tokens'] = [0 for _ in range(batch_size)]
    
    if return_output_length:
        output_dict['output_lengths'] = sequence_lengths+1
    if return_log_probs:
        output_dict['cum_log_probs'] = cum_log_probs
        output_dict['output_log_probs'] = output_log_probs
    return output_dict
