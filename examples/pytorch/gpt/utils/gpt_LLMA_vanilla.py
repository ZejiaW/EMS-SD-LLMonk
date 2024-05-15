import torch, time
from typing import Optional
import numpy as np
from . import comm
from . import profiler

def gpt_LLMA(
        gpt,
        check_length: int,
        copy_length: int,
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
    assert gpt.weight is not None, 'Please call load() first to initialize weights.'

    ENABLE_LLMA = copy_length!=0
    input_token_ids = input_token_ids.type(torch.int32).to(gpt.device)
    input_lengths = input_lengths.type(torch.int32).to(gpt.device)

    batch_size = len(input_token_ids)
    max_input_length = input_token_ids.shape[-1]
    max_seq_length = max_input_length + gen_length*(copy_length+1)
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
    sequence_lengths = (max_input_length - 1) * torch.ones_like(input_lengths)

    if return_log_probs or beam_width > 1:
        cum_log_probs = torch.zeros(batch_size * beam_width, device=device)
        output_log_probs = torch.zeros((gen_length, batch_size * beam_width), device=device)
    else:
        cum_log_probs = None
        output_log_probs = None

    # Contiguous buffer for each decode_op step, it will be transposed tensor for the final output.
    global_position_ids = torch.arange(max_seq_length, dtype=torch.int32, device=device).unsqueeze(1).repeat(1,batch_size)
    output_token_ids = torch.zeros(
        (max_seq_length, batch_size * beam_width), dtype=torch.int32, device=device)
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
    else:
        # Dummy input_embeds
        input_embeds = torch.empty(
            size=(batch_size * beam_width, max_input_length, gpt.context_decoder.hidden_size),
            dtype=gpt.context_decoder.dtype,
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
    profiler.stop('ft-context-decoder')

    torch.cuda.synchronize()
    context_time = time.time()*1000
    wall_time_list = [context_time-start_time]

    # torch.Size([48, 2, 25, 8, 758, 8]) 
    # layers,bs,n_head, 8, lengths, 8
    
    if ENABLE_LLMA:
        lookup_table = [dict() for _ in range(batch_size)]
        # lookup table
        for idx in range(batch_size):
            cur_length = input_lengths[idx].item()
            input_ids_list = input_token_ids[idx][:cur_length].tolist()
            for index in range(cur_length-check_length-copy_length+1):
                lookup_table[idx][tuple(input_ids_list[index:index+check_length])] = index + check_length
    # for step in range(max_input_length, max_seq_length):
    first_context_decoding = True
    inference_steps = torch.zeros(batch_size, device=device, dtype=torch.int)
    generated_tokens = torch.zeros((batch_size), dtype=torch.int32, device=input_embeds.device)
    for step in range(max_input_length, max_seq_length):
        inference_steps += (1-finished.int())
        for ite in range(num_local_batches):
            draft_start_time, draft_end_time, decoder_end_time, post_process_end_time = 0,0,0,0
            torch.cuda.synchronize()
            draft_start_time = time.time()*1000
            draft_sequence_lengths = sequence_lengths.clone()+1
            # predict
            token_nums_per_sample = torch.ones((batch_size), dtype=torch.int32, device=input_embeds.device)
            if ENABLE_LLMA and not first_context_decoding:
                for idx in range(batch_size):
                    if finished[idx]:
                        draft_sequence_lengths[idx] = sequence_lengths[idx]
                        token_nums_per_sample[idx] = 0
                        continue
                    cur_length = sequence_lengths[idx].item()+1
                    
                    def find_match_tuple(output_token_ids, masked_tokens, cur_length, check_length, idx):
                        # output_token_ids[cur_length-check_length:cur_length, idx]
                        match_tokens = []
                        while len(match_tokens) < check_length:
                            if not masked_tokens[idx, cur_length-1]:
                                match_tokens.append(output_token_ids[cur_length-1, idx].item())
                            cur_length -= 1
                        return tuple(match_tokens[::-1])
                    copy_index = lookup_table[idx].get(find_match_tuple(output_token_ids, masked_tokens, cur_length, check_length, idx), None)
                    cur_copy_length = min(copy_length, max_seq_length-cur_length-1)
                    if copy_index is not None:
                        output_token_ids[cur_length:cur_length+cur_copy_length, idx] = output_token_ids[copy_index:copy_index+cur_copy_length, idx]
                        global_position_ids[cur_length:cur_length+cur_copy_length, idx] = torch.arange(input_lengths[idx].item()+generated_tokens[idx].item(),input_lengths[idx].item()+generated_tokens[idx].item()+cur_copy_length)
                        draft_sequence_lengths[idx] += cur_copy_length
                        token_nums_per_sample[idx] = cur_copy_length + 1
                    else:
                        global_position_ids[cur_length:cur_length+cur_copy_length, idx] = torch.arange(input_lengths[idx].item()+generated_tokens[idx].item(),input_lengths[idx].item()+generated_tokens[idx].item()+cur_copy_length)
            # The indices of the current local batch-beam.
            bbidx = range(
                ite * local_batch_size * beam_width,
                min((ite + 1) * local_batch_size * beam_width, batch_size * beam_width))
            
            token_nums_per_sample_min = token_nums_per_sample.min().item()
            token_nums_per_sample_max = token_nums_per_sample.max().item()

            if first_context_decoding:
                hidden_states = last_token_hidden_states[bbidx, ...]
                first_context_decoding = False
            else:
                if comm.is_pipeline_group_first():
                    # LLM model inference prepare
                    if ENABLE_LLMA and not (token_nums_per_sample_max == 1 and token_nums_per_sample_min == 1):
                        global_position_idx = sequence_lengths[0].item()

                        token_nums_per_sample[:] = token_nums_per_sample_max
                        input_tokens = output_token_ids[global_position_idx:global_position_idx+token_nums_per_sample_max, bbidx]
                        input_tokens = input_tokens.T.flatten()

                        position_ids = global_position_ids[global_position_idx:global_position_idx+token_nums_per_sample_max, bbidx]
                        position_ids = position_ids.T.flatten()
                    else:
                        input_tokens = output_token_ids[sequence_lengths.cpu().numpy(), bbidx]
                        position_ids = global_position_ids[sequence_lengths.cpu().numpy(), bbidx]
                    position_ids = torch.clamp(position_ids, max=2046)
                    input_embeds = gpt.word_embedding(input_tokens)
                    if gpt.position_encoding is not None:
                        input_embeds += gpt.position_encoding(position_ids)
                    if gpt.pre_decoder_layernorm is not None:
                        input_embeds = gpt.pre_decoder_layernorm(input_embeds)
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
                profiler.start('ft-decode')
                # argmax
                predict_tokens = logits[:,:vocab_size].argmax(-1).type(torch.int32)
                if ENABLE_LLMA and not (token_nums_per_sample_max == 1 and token_nums_per_sample_min == 1):
                    output_token_ids[global_position_idx+1:global_position_idx+1+token_nums_per_sample_max, bbidx] = predict_tokens.reshape(batch_size, -1).T
                    global_position_ids[global_position_idx+token_nums_per_sample_max, bbidx] = input_lengths+generated_tokens+token_nums_per_sample_max-1
                    # verify
                    
                    accept_lengths = np.zeros(batch_size, dtype=int)
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
                        accept_lengths[idx] = accept_length
                        batch_offset += token_nums_item
                        if predict_tokens_idx[accept_length-1] == eos_token_id:
                            finished[idx] = True
                    max_accept_length = accept_lengths.max()
                    for idx in range(batch_size):
                        if accept_lengths[idx] != max_accept_length and not finished[idx]:
                            masked_tokens[idx, global_position_idx+accept_lengths[idx]:global_position_idx+max_accept_length] = True
                            output_token_ids[global_position_idx+accept_lengths[idx]:global_position_idx+max_accept_length+1, idx] = predict_tokens.reshape(batch_size, -1)[idx][accept_lengths[idx]-1]
                            global_position_ids[global_position_idx+accept_lengths[idx]:global_position_idx+max_accept_length+1, idx] = input_lengths[idx]+generated_tokens[idx]+accept_lengths[idx]-1
                        generated_tokens[idx] += accept_lengths[idx]
                        if generated_tokens[idx]>=gen_length:
                            finished[idx] = True
                        
                    sequence_lengths += max_accept_length
                else:
                    sequence_lengths += 1
                    output_token_ids[sequence_lengths.cpu().numpy(), bbidx] = predict_tokens.type(torch.int32)
                    global_position_ids[sequence_lengths.cpu().numpy(), bbidx] = input_lengths+generated_tokens
                    generated_tokens += (1-finished.int())

                    finished = finished  | (predict_tokens==eos_token_id)
                    finished = finished  | (generated_tokens>=gen_length)
                profiler.stop('ft-decode')
                torch.cuda.synchronize()
                post_process_end_time = time.time()*1000
                if step == max_input_length:
                    wall_time_list.extend([decoder_end_time-draft_start_time, post_process_end_time-decoder_end_time])
                else:
                    wall_time_list.extend([draft_end_time-draft_start_time, decoder_end_time-draft_end_time, post_process_end_time-decoder_end_time])
        if finished.all():
            break
    torch.cuda.synchronize()
    end_time = time.time()

    global_start_idx = input_lengths.max().item()
    global_end_idx = sequence_lengths[0].item()
    pad_tokens = [0 for _ in range(batch_size)]
    for idx in range(batch_size):
        start_idx = input_lengths[idx].item()
        for ori_idx in range(global_start_idx, global_end_idx):
            if not masked_tokens[idx, ori_idx]:
                output_token_ids[start_idx, idx] = output_token_ids[ori_idx, idx]
                if output_token_ids[ori_idx, idx] == eos_token_id or start_idx - input_lengths[idx].item()+1 >= gen_length:
                    break
                start_idx += 1
            else:
                pad_tokens[idx] += 1
        sequence_lengths[idx] = start_idx


    # Transpose (L, batch, beam) -> (batch, beam, L)
    output_token_ids = output_token_ids.view(-1, batch_size, beam_width).permute(1, 2, 0)

    # Outputs
    output_dict = dict(output_token_ids=output_token_ids)
    output_dict['inference_steps'] = inference_steps
    output_dict['wall_time'] = wall_time_list
    output_dict['pad_tokens'] = pad_tokens
    
    if return_output_length:
        output_dict['output_lengths'] = sequence_lengths+1
    if return_log_probs:
        output_dict['cum_log_probs'] = cum_log_probs
        output_dict['output_log_probs'] = output_log_probs
    return output_dict
