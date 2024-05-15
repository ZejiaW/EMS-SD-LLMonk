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
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/GPT/HF/gpt2-xl/c-models')
    parser.add_argument('--hf_model_name', type=str,
                        default='facebook/opt-350m')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
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
        '--use_gpt_decoder_ops', action='store_true',
        help='Use separate decoder FT operators instead of end-to-end model op.')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='max batch size.')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--decode_len', type=int, default=5)
    parser.add_argument('--save_root', type=str, default="../outputs")
    parser.add_argument('--name', type=str, default="nona")
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat_times', type=int, default=32)

    args = parser.parse_args()
    
    np.random.seed(1) # rouge score use sampling to compute the score

    comm.initialize_model_parallel(args.tensor_para_size, args.pipeline_para_size)
    rank = comm.get_rank()

    ft_model_location = args.ft_model_location
    hf_model_name = args.hf_model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

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

    max_seq_len = hf_config['max_position_embeddings']

    vocab_size = hf_config['vocab_size']
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    lib_path = args.lib_path
    ckpt_path = os.path.join(ft_model_location, f'{tensor_para_size}-gpu')

    assert args.use_gpt_decoder_ops

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
        int8_mode=False,
        inference_data_type=args.data_type,
        weights_data_type=args.weights_data_type,
        use_fp32_to_compute_logit=False)
    gpt.load(ckpt_path, args.data_type)
    
    stat_dict = {
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "decode_len": args.decode_len,
        "model": args.hf_model_name,
        "data_type": args.data_type,
        "times": []
    }
    for i in trange(args.warmup+args.repeat_times):
        attention_mask = torch.ones(
        (args.input_len, args.input_len), dtype=torch.bool, device=gpt.device)\
        .tril().unsqueeze(0).tile(args.batch_size, 1, 1).to(gpt.dtype)
        
        input_embeds = torch.randn((args.batch_size, args.input_len, hf_config['hidden_size']), dtype=torch.float16, device=gpt.device)
        input_lengths = torch.ones(args.batch_size).type(torch.int32).to(gpt.device)
        memory_length = args.input_len + args.decode_len

        use_shared_contexts = (gpt.shared_contexts_ratio > 0.) and (args.input_len >= 1) and (args.batch_size > 1)
        batch_to_compact, compact_to_batch = None, None
        
        torch.cuda.synchronize()
        start_time = time.time()*1000
        # context decode
        _, k_cache, v_cache, last_token_hidden_states = gpt.context_decoder.forward(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            memory_length=memory_length,
            batch_to_compact_index=batch_to_compact,
            compact_index=compact_to_batch)
        torch.cuda.synchronize()
        context_time = time.time()*1000
        time_list = [context_time-start_time]

        sequence_lengths = torch.ones((args.batch_size), dtype=torch.int32, device=input_embeds.device)
        pad_lengths = torch.zeros((args.batch_size), dtype=torch.int32, device=input_embeds.device)
        finished = torch.zeros_like(sequence_lengths).bool()
        masked_tokens = gpt.generate_pad_mask(input_lengths, memory_length)
        
        for input_len in [1, args.decode_len]:
            input_embeds = torch.randn((args.batch_size*input_len, hf_config['hidden_size']), dtype=torch.float16, device=gpt.device)
            token_nums_per_sample = torch.ones((args.batch_size), dtype=torch.int32, device=input_embeds.device) * input_len
            torch.cuda.synchronize()
            decode_start_time = time.time()*1000
            hidden_states = gpt.decoder.forward(
                        max_input_length=args.input_len,
                        step=0,
                        ite=0,
                        input_embeds=input_embeds,
                        sequence_lengths=sequence_lengths,
                        key_cache=k_cache,
                        value_cache=v_cache,
                        finished=finished,
                        total_padding_tokens=pad_lengths,
                        cache_indirection=None,
                        masked_tokens=masked_tokens,
                        token_nums_per_sample=token_nums_per_sample)
            torch.cuda.synchronize()
            decode_end_time = time.time()*1000
            time_list.append(decode_end_time-decode_start_time)
        if i >= args.warmup:
            stat_dict['times'].append(time_list)

    os.makedirs(f"{args.save_root}/", exist_ok=True)
    with open(f"{args.save_root}/{args.name}.jsonl", 'w') as f:
        json.dump(stat_dict, f, ensure_ascii=False)
    print("TEST END, save:", f"{args.save_root}/{args.name}.jsonl")

if __name__ == '__main__':
    main()
