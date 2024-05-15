# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from utils import comm
from tqdm import trange

from utils import gpt_decoder
from utils import profiler
import json
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/GPT/HF/gpt2-xl/c-models')
    parser.add_argument('--hf_model_name', type=str,
                        default='facebook/opt-350m')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument(
        '--int8_mode', type=int, default=0, choices=[0, 1],
        help='The level of quantization to perform.'
             ' 0: No quantization. All computation in data_type'
             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    parser.add_argument(
        '--use_gpt_decoder_ops', action='store_true',
        help='Use separate decoder FT operators instead of end-to-end model op.')
    parser.add_argument(
        '--use_fp32_to_compute_logit', action='store_true',
        help='Use FP32 data type for computing logit values when using gpt decoder ops. '
             'FT end-to-end GPT op always uses FP32 data type when computing logit.')
    parser.add_argument(
        '--rougeLsum_threshold', type=float, default=None,
        help='Threshold of FT rougeLsum score')
    parser.add_argument(
        '--verbose', action='store_true', help='Print all summary result.')
    parser.add_argument(
        '--max_batch_size', type=int, default=1, help='max batch size.')
    parser.add_argument('--output_len', type=int, default=128)
    parser.add_argument('--save_root', type=str, default="../outputs")
    parser.add_argument('--name', type=str, default="nona")
    parser.add_argument('--method', type=str, default="baseline")
    parser.add_argument('--check_length', type=int, default=2)
    parser.add_argument('--copy_length', type=int, default=7)
    parser.add_argument('--draft_ft_model_location', type=str,
                        default='/models/GPT/HF/gpt2-xl/c-models')
    parser.add_argument('--draft_hf_model_name', type=str,
                        default='facebook/opt-350m')
    parser.add_argument('--draft_steps_max', type=int, default=4)

    args = parser.parse_args()
    if args.method == "baseline":
        pass
    elif args.method == "LLMA_vanilla":
        from utils.gpt_LLMA_vanilla import gpt_LLMA
    elif args.method == "LLMA_EMS":
        from utils.gpt_LLMA_EMS import gpt_LLMA
    elif args.method == "LLMA_EMS_ablation_unpad_inputs":
        from utils.gpt_LLMA_EMS_ablation_unpad_inputs import gpt_LLMA
    elif args.method == "LLMA_EMS_ablation_unpad_kv_cache":
        from utils.gpt_LLMA_EMS_ablation_unpad_kv_cache import gpt_LLMA
    elif args.method == "SD_vanilla":
        from utils.gpt_specutive_decoding_vanilla import gpt_speculative_decoding
    elif args.method == "SD_EMS":
        from utils.gpt_specutive_decoding_EMS import gpt_speculative_decoding
    else:
        raise NotImplemented
    

    np.random.seed(1) # rouge score use sampling to compute the score
    comm.initialize_model_parallel(args.tensor_para_size, args.pipeline_para_size)
    rank = comm.get_rank()

    if rank == 0:
        os.makedirs(f"{args.save_root}", exist_ok=True)
        f = open(f"{args.save_root}/{args.name}.jsonl", 'w')

    summarize = args.summarize
    test_hf = args.test_hf
    ft_model_location = args.ft_model_location
    hf_model_name = args.hf_model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_cnn = load_from_disk("../datasets/cnn_dailymail/")

    hf_config = vars(AutoConfig.from_pretrained(hf_model_name))

    head_num = hf_config['num_attention_heads']
    layer_num = hf_config['num_hidden_layers']
    start_id = hf_config['bos_token_id']
    end_id = hf_config['eos_token_id']
    size_per_head = hf_config['hidden_size'] // head_num

    # opt specific params: some are fixed
    layernorm_eps = 1e-5
    layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
    activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L498
    # has post decoder layernorm when layernorm_type is pre layernorm
    has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'

    top_k = 1
    output_len = args.output_len
    top_p = 0.0
    temperature = 0
    max_seq_len = hf_config['max_position_embeddings']
    max_batch_size = args.max_batch_size
    repetition_penalty = 1
    random_seed = 0
    vocab_size = hf_config['vocab_size']
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    lib_path = args.lib_path
    ckpt_path = os.path.join(ft_model_location, f'{tensor_para_size}-gpu')

    if rank == 0:
        print(f"top_k: {top_k}")
        print(f"top_p: {top_p}")
        print(f"int8_mode: {args.int8_mode}")
        print(f"temperature: {temperature}")
        print(f"max_seq_len: {max_seq_len}")
        print(f"max_batch_size: {max_batch_size}")
        print(f"repetition_penalty: {repetition_penalty}")
        print(f"vocab_size: {vocab_size}")
        print(f"tensor_para_size: {tensor_para_size}")
        print(f"pipeline_para_size: {pipeline_para_size}")
        print(f"lib_path: {lib_path}")
        print(f"ckpt_path: {ckpt_path}")
        print(f"hf_config: {hf_config}")

    infer_decode_args = dict(
        beam_width=1,
        top_k=top_k * torch.ones(max_batch_size, dtype=torch.int32),
        top_p=top_p * torch.ones(max_batch_size, dtype=torch.float32),
        temperature=temperature * torch.ones(max_batch_size, dtype=torch.float32),
        repetition_penalty=repetition_penalty * torch.ones(max_batch_size, dtype=torch.float32),
        random_seed=random_seed * torch.ones(max_batch_size, dtype=torch.int64)
    )

    if not args.use_gpt_decoder_ops:
        assert False, "only support use_gpt_decoder_ops now"
        gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                          max_seq_len, tensor_para_size, pipeline_para_size, lib_path,
                          inference_data_type=args.data_type,
                          layernorm_eps=layernorm_eps,
                          layernorm_type=layernorm_type,
                          activation_type=activation_type,
                          has_post_decoder_layernorm=has_post_decoder_layernorm,
                          int8_mode=args.int8_mode,
                          weights_data_type=args.weights_data_type)
        if not gpt.load(ckpt_path=ckpt_path):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    else:
        gpt = gpt_decoder.Gpt(
            num_heads=head_num,
            size_per_head=size_per_head,
            num_layers=layer_num,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
            tensor_para_size=tensor_para_size,
            pipeline_para_size=pipeline_para_size,
            lib_path=lib_path,
            max_seq_len=max_seq_len,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            activation_type=activation_type,
            has_post_decoder_layernorm=has_post_decoder_layernorm,
            int8_mode=args.int8_mode,
            inference_data_type=args.data_type,
            weights_data_type=args.weights_data_type,
            use_fp32_to_compute_logit=args.use_fp32_to_compute_logit)
        gpt.load(ckpt_path, args.data_type)
        if "SD" in args.method:
            draft_steps_max = args.draft_steps_max

            draft_ft_model_location = args.draft_ft_model_location
            draft_ckpt_path = os.path.join(draft_ft_model_location, f'{tensor_para_size}-gpu')
            draft_hf_model_name = args.draft_hf_model_name
            draft_hf_config = vars(AutoConfig.from_pretrained(draft_hf_model_name))

            draft_head_num = draft_hf_config['num_attention_heads']
            draft_layer_num = draft_hf_config['num_hidden_layers']
            draft_size_per_head = draft_hf_config['hidden_size'] // draft_head_num

            draft_layernorm_type = 'pre_layernorm' if draft_hf_config['do_layer_norm_before'] else 'post_layernorm'
            draft_activation_type = 'Relu' if draft_hf_config['activation_function'] == 'relu' else 'Gelu'
            
            draft_has_post_decoder_layernorm = draft_layernorm_type == 'pre_layernorm'
            draft_gpt = gpt_decoder.Gpt(
                num_heads=draft_head_num,
                size_per_head=draft_size_per_head,
                num_layers=draft_layer_num,
                vocab_size=vocab_size,
                start_id=start_id,
                end_id=end_id,
                tensor_para_size=tensor_para_size,
                pipeline_para_size=pipeline_para_size,
                lib_path=lib_path,
                max_seq_len=max_seq_len,
                layernorm_eps=layernorm_eps,
                layernorm_type=draft_layernorm_type,
                activation_type=draft_activation_type,
                has_post_decoder_layernorm=draft_has_post_decoder_layernorm,
                int8_mode=args.int8_mode,
                inference_data_type=args.data_type,
                weights_data_type=args.weights_data_type,
                use_fp32_to_compute_logit=args.use_fp32_to_compute_logit)
            draft_gpt.load(draft_ckpt_path, args.data_type)

    if (test_hf and summarize) or not summarize:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        model.cuda()
        if args.data_type == 'fp16':
            model.half()
        elif args.data_type == 'bf16':
            model.bfloat16()

    def summarize_ft_e2e(datapoint):
        if summarize:
            line = datapoint['article'] + ' TL;DR: '
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        if summarize:
            line_encoded = line_encoded[:, -(max_seq_len-output_len-1):]
        else:
            line_encoded = line_encoded[:, -768:]
        line_encoded = line_encoded.type(torch.int32)

        with torch.no_grad():
            output, ft_output_len = gpt(line_encoded, torch.IntTensor([len(line_encoded[0])]),
                                        output_len,
                                        return_output_length=True,
                                        **infer_decode_args)

        tokens = output[0][0][len(line_encoded[0]):ft_output_len[0]].cpu().numpy()

        output_lines = tokenizer.decode(output[0][0][len(line_encoded[0]):ft_output_len[0]])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    def summarize_ft_sep(datapoint, data_idx):
        def process_input_text(text):
            return text.strip().replace(" n't", "n't")
        if summarize:
            lines = [process_input_text(line + ' TL;DR: ') for line in datapoint['article']]
        else:
            lines = [process_input_text(line) for line in datapoint['article']]
        tokenizer_input = tokenizer(lines, return_tensors='pt', padding=True)
        line_encoded = tokenizer_input['input_ids']
        input_lengths = tokenizer_input['attention_mask'].sum(1)
        
        if line_encoded.shape[1] > max_seq_len-output_len-1:
            for idx, input_length in enumerate(input_lengths):
                if input_length > max_seq_len-output_len-1:
                    cur_start_idx = input_length.item()-(max_seq_len-output_len-1)
                    line_encoded[idx][:(max_seq_len-output_len-1)] = line_encoded[idx][cur_start_idx:input_length.item()].clone()
                    input_lengths[idx] = max_seq_len-output_len-1
            line_encoded = line_encoded[:,:(max_seq_len-output_len-1)].contiguous()
        line_encoded = line_encoded.type(torch.int32).to(gpt.device)
        input_lengths = input_lengths.type(torch.int32).to(gpt.device)
        with torch.no_grad():
            profiler.start('ft')
            if args.method == "baseline":
                output_dict = gpt.generate(input_token_ids=line_encoded,
                                        input_lengths=input_lengths,
                                        gen_length=output_len,
                                        eos_token_id=tokenizer.eos_token_id,
                                        return_output_length=True,
                                        **infer_decode_args)
            elif "LLMA" in args.method:
                output_dict = gpt_LLMA(gpt, check_length=args.check_length, copy_length=args.copy_length, vocab_size=50272,
                                    input_token_ids=line_encoded,
                                       input_lengths=input_lengths,
                                       gen_length=output_len,
                                       eos_token_id=tokenizer.eos_token_id,
                                       return_output_length=True,
                                       **infer_decode_args)
            elif "SD" in args.method:
                output_dict = gpt_speculative_decoding(gpt, draft_gpt, draft_steps_max,
                                        vocab_size,
                                        input_token_ids=line_encoded,
                                        input_lengths=input_lengths,
                                        gen_length=output_len,
                                        eos_token_id=tokenizer.eos_token_id,
                                        return_output_length=True,
                                        **infer_decode_args)
            profiler.stop('ft')
        output_token_ids = output_dict['output_token_ids']
        output_lengths = output_dict['output_lengths']
        batch_input_lines = []
        batch_output_lines = []
        batch_tokens = []
        for idx in range(output_token_ids.shape[0]):
            tokens = output_token_ids[idx, 0, :output_lengths[idx]]

            output_lines = tokenizer.decode(tokens[input_lengths[idx]:])
            output_lines = ".".join(output_lines.split('.')[:4]) + "."
            batch_output_lines.append(output_lines)
            batch_tokens.append(tokens[input_lengths[idx]:].cpu().tolist())

            input_lines = tokenizer.decode(tokens[:input_lengths[idx]])
            batch_input_lines.append(input_lines)
        
        # write results
        if rank == 0 and data_idx is not None: 
            stat_dict = {
                "wall_time": output_dict['wall_time'],
                "new_tokens": (output_lengths-input_lengths).cpu().tolist(),
                "inference_steps": output_dict['inference_steps'].cpu().tolist(),
                # "input_text": batch_input_lines,
                "output_text": batch_output_lines,
                "output_tokens": batch_tokens,
                "data_idx": data_idx,
                "input_lengths": input_lengths.cpu().tolist(),
                "pad_tokens": output_dict['pad_tokens']
            }
            json.dump(stat_dict, f, ensure_ascii=False)
            f.write("\n")
        
        return batch_output_lines, batch_tokens

    summarize_ft = summarize_ft_e2e if not args.use_gpt_decoder_ops else summarize_ft_sep

    def summarize_hf(datapoint):
        if summarize:
            line = datapoint['article'] + ' TL;DR: '
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        line_encoded = line_encoded[:,:(max_seq_len-output_len-1)].contiguous()
        # line_encoded = line_encoded.to(device_hf)
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=len(line_encoded[0]) + output_len,
                                    top_k=top_k,
                                    temperature=temperature,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)

        tokens = output[0][len(line_encoded[0]):].cpu().numpy()
        output_lines = tokenizer.decode(output[0][len(line_encoded[0]):])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens
    # warm up 
    if summarize:
        datapoint = dataset_cnn['test'][:args.max_batch_size]
        summary, _ = summarize_ft(datapoint, None)
        if rank == 0:
            print('---------------------------------------------------------')
            print('FT Generated : ')
            print(' Article : ', datapoint['article'])
            print('\n Highlights : ', datapoint['highlights'])
            print('\n Summary : ', summary)
            print('---------------------------------------------------------')

        if test_hf:
            summary, _ = summarize_hf(dataset_cnn['test'][0])
            if rank == 0:
                print('---------------------------------------------------------')
                print('HF Generated : ')
                print(' Article : ', datapoint['article'][0])
                print('\n Highlights : ', datapoint['highlights'][0])
                print('\n Summary : ', summary)
                print('---------------------------------------------------------')

    if summarize:
        metric_ft = load_metric("../datasets/rouge.py")
        metric_hf = load_metric("../datasets/rouge.py")
    else:
        tokens = []

    ft_time = 0.0
    hf_time = 0.0
    selected_ids = pd.read_csv("../datasets/cnn_daiymail_selected_ids.csv", header=None)[0].tolist()
    assert len(selected_ids) >= args.num_samples
    for data_point_idx in trange(0, args.num_samples+1-args.max_batch_size, args.max_batch_size):
        if 1:
            datapoint = dataset_cnn['test'][selected_ids[data_point_idx:data_point_idx+args.max_batch_size]]

            start_time = datetime.now()
            summary_ft, tokens_ft = summarize_ft(datapoint, selected_ids[data_point_idx:data_point_idx+args.max_batch_size])
            stop_time = datetime.now()
            ft_time += (stop_time - start_time).total_seconds()
            if (test_hf and summarize) or not summarize:
                summary_hf_list = []
                for idx in range(data_point_idx, data_point_idx+args.max_batch_size):
                    start_time = datetime.now()
                    summary_hf, tokens_hf = summarize_hf(dataset_cnn['test'][idx])
                    stop_time = datetime.now()
                    hf_time += (stop_time - start_time).total_seconds()
                    summary_hf_list.append(summary_hf)

            if rank == 0:
                if summarize:
                    metric_ft.add_batch(predictions=summary_ft, references=datapoint['highlights'])
                    if test_hf:
                        metric_hf.add_batch(predictions=summary_hf_list, references=datapoint['highlights'])
                else:
                    tokens.append((tokens_ft, tokens_hf))
                if args.verbose:
                    print('-' * 100)
                    print('FT Summary:', summary_ft)
                    if test_hf:
                        print('HF Summary:', summary_hf_list)
        # except:
        #     print('Error with datapoint : ', data_point_idx)

    def compute_exact_match(tokens, n_tokens=[1, 10, 25, 50, 100, 150, 200, 250]):
        em_metrics = []
        for t in n_tokens:
            errors = 0
            total = 0
            for ft_tokens, hf_tokens in tokens:
                if len(ft_tokens) > t and len(hf_tokens) > t:
                    total = total + 1
                    if not np.array_equal(ft_tokens[:t], hf_tokens[:t]):
                        errors = errors + 1

            if total > 0:
                print(f"{t}-token exact match acc: {100*(1-errors/total):.2f}")
                em_metrics.append(1 - errors / total)
            else:
                em_metrics.append(np.nan)

        return em_metrics

    if rank == 0:
        if summarize:
            computed_metrics_ft = metric_ft.compute()

            if test_hf:
                computed_metrics_hf = metric_hf.compute()

                print(f'Hugging Face (total latency: {hf_time} sec)')
                for key in computed_metrics_hf.keys():
                    print(f'{key} : {computed_metrics_hf[key].mid[2]*100}')

            print(f'Faster Transformers (total latency: {ft_time} sec)')
            for key in computed_metrics_ft.keys():
                print(f'{key} : {computed_metrics_ft[key].mid[2]*100}')
            if args.rougeLsum_threshold is not None:
                assert computed_metrics_ft["rougeLsum"].mid[2]*100 >= args.rougeLsum_threshold, "[INFO] TEST FAIL !"
                print(f"[INFO] TEST PASS !")
        else:
            em_metrics = compute_exact_match(tokens)
        profiler.summary()


if __name__ == '__main__':
    main()
