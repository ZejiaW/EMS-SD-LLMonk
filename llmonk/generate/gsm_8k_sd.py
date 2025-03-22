import sys
import os
from datetime import datetime

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..'                  
        )
    )
)

import torch
import argparse
import json
import yaml
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, Any, TextIO

from functools import partial

from examples.pytorch.gpt.utils.gpt_decoder import Gpt
from examples.pytorch.gpt.utils.gpt_specutive_decoding_EMS import gpt_speculative_decoding
from examples.pytorch.gpt.utils import profiler

def save_yaml(path: Path, data, sort_keys=True):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)

class GSM8KConfig:
    def __init__(self, args: argparse.Namespace):
        # Model configuration
        self.target_ckpt = Path(args.target_ckpt)
        self.draft_ckpt = Path(args.draft_ckpt)
        self.tensor_para_size = args.tensor_para_size
        self.pipeline_para_size = args.pipeline_para_size
        self.lib_path = args.lib_path
        self.int8_mode = args.int8_mode
        self.inference_data_type = args.inference_data_type
        self.weights_data_type = args.weights_data_type
        self.draft_steps_max = args.draft_steps_max
        
        # Load model configs
        self.target_hf_model_name = args.target_hf_model_name
        self.draft_hf_model_name = args.draft_hf_model_name
        self.target_config = vars(AutoConfig.from_pretrained(args.target_hf_model_name))
        self.draft_config = vars(AutoConfig.from_pretrained(args.draft_hf_model_name))

        # data configs
        self.num_few_shot = args.num_few_shot
        self.limit = args.limit
        self.offset = args.offset
        self.stride = args.stride
        
        # Generation parameters
        self.save_dir = Path(args.save_dir)
        self.num_samples = args.num_samples
        self.max_batch_size = args.max_batch_size
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.random_seed = args.random_seed
        self.output_len = args.output_len

        # method and name
        self.method = args.method
        self.name = args.name

        # Create output directory
        # self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = self.save_dir / self.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

def get_few_shot_prompt(item):
    return "".join(
        f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        for f in item["few_shot_items"]
    )

def load_opt_model(ckpt_path: str, config: dict, tensor_para_size: int, lib_path: str, dtype: str = "fp16", int8_mode: int = 0):
    """Load OPT C-model"""
    ckpt_path = ckpt_path / f"{tensor_para_size}-gpu"
    model = Gpt(
        num_heads=config['num_attention_heads'],
        size_per_head=config['hidden_size'] // config['num_attention_heads'],
        num_layers=config['num_hidden_layers'],
        vocab_size=config['vocab_size'],
        start_id=config['bos_token_id'],
        end_id=config['eos_token_id'],
        tensor_para_size=tensor_para_size,
        pipeline_para_size=1,
        lib_path=lib_path,
        max_seq_len=config['max_position_embeddings'],
        layernorm_eps=1e-5,
        layernorm_type='pre_layernorm' if config['do_layer_norm_before'] else 'post_layernorm',
        activation_type='Relu' if config['activation_function'] == 'relu' else 'Gelu',
        has_post_decoder_layernorm=config['do_layer_norm_before'],
        int8_mode=int8_mode,
        inference_data_type=dtype,
        weights_data_type=dtype
    )
    model.load(ckpt_path, dtype)
    return model

def run_inference(item: Dict, 
                  config: GSM8KConfig,
                  target_gpt: Gpt,
                  draft_gpt: Gpt,
                  tokenizer: AutoTokenizer,
                  log_file: TextIO) -> None:
    
    outpath = config.save_dir / "results"
    outpath.mkdir(parents=True, exist_ok=True)
    outpath = outpath / f"{item['id']}.yaml"
    if outpath.exists():
        return

    infer_decode_args = dict(
        beam_width=1,
        top_k=config.top_k * torch.ones(config.max_batch_size, dtype=torch.int32),
        top_p=config.top_p * torch.ones(config.max_batch_size, dtype=torch.float32),
        temperature=config.temperature * torch.ones(config.max_batch_size, dtype=torch.float32),
        # repetition_penalty=config.repetition_penalty * torch.ones(config.max_batch_size, dtype=torch.float32),
        repetition_penalty=torch.full((config.max_batch_size,), config.repetition_penalty, dtype=torch.float32),
        random_seed=torch.randint(10000, (config.max_batch_size,), dtype=torch.int64)
    )
    max_seq_len = config.target_config['max_position_embeddings']
    output_len = config.output_len


    # Prepare input
    few_shot_prompt = get_few_shot_prompt(item)
    prompt = [few_shot_prompt + f"Question: {item['question']}\nAnswer:" for _ in range(config.max_batch_size)]

    # Tokenize input
    tokenizer_input = tokenizer(prompt, return_tensors='pt', padding=True)
    line_encoded = tokenizer_input['input_ids']
    input_lengths = tokenizer_input['attention_mask'].sum(1)
    # print(f"line_encoded: {line_encoded.shape}")

    # Truncate input if it's too long
    # print(f"line_encoded.shape[1]: {line_encoded.shape[1]}")
    # print(f"max_seq_len: {max_seq_len}")
    # print(f"output_len: {output_len}")
    # print(f"max_seq_len - output_len - 1: {max_seq_len - output_len - 1}")
    if line_encoded.shape[1] > max_seq_len - output_len - 1:
        for idx, input_length in enumerate(input_lengths):
            if input_length > max_seq_len - output_len - 1:
                cur_start_idx = input_length.item() - (max_seq_len - output_len - 1)
                line_encoded[idx][:(max_seq_len - output_len - 1)] = line_encoded[idx][cur_start_idx:input_length.item()].clone()
                input_lengths[idx] = max_seq_len - output_len - 1
        line_encoded = line_encoded[:,:(max_seq_len - output_len - 1)].contiguous()
        # print(f"line_encoded: {line_encoded.shape}")
    line_encoded = line_encoded.type(torch.int32).to(target_gpt.device)
    input_lengths = input_lengths.type(torch.int32).to(target_gpt.device)

    # Run inference
    samples = []
    assert config.num_samples % config.max_batch_size == 0
    with torch.no_grad():
        for _ in trange(config.num_samples // config.max_batch_size, desc=f"Item {item['id']}"):
            if config.method == "SD_EMS":
                # Run speculative decoding
                profiler.start('sd')
                output_dict = gpt_speculative_decoding(
                    target_gpt,
                    draft_gpt,
                    config.draft_steps_max,
                    vocab_size=config.target_config['vocab_size'],
                    input_token_ids=line_encoded,
                    input_lengths=input_lengths,
                    gen_length=output_len,
                    eos_token_id=tokenizer.eos_token_id,
                    return_output_length=True,
                    stop_words_list=torch.tensor([[tokenizer.eos_token_id]]*config.max_batch_size, dtype=torch.int32),
                    sequence_limit_lengths=(input_lengths + output_len).clamp(max=config.target_config['max_position_embeddings']),
                    **infer_decode_args
                )
                profiler.stop('sd')
            elif config.method == "Vanilla":
                profiler.start('vanilla')
                output_dict = target_gpt.generate(input_token_ids=line_encoded,
                                                    input_lengths=input_lengths,
                                                    gen_length=output_len,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    return_output_length=True,
                                                    **infer_decode_args)
                profiler.stop('vanilla')
            else:
                raise ValueError(f"Invalid method: {config.method}")
            # Decode output
            output_token_ids = output_dict['output_token_ids']
            output_lengths = output_dict['output_lengths']
            batch_input_lines = []
            batch_output_lines = []
            batch_tokens = []
            for idx in range(output_token_ids.shape[0]):
                tokens = output_token_ids[idx, 0, :output_lengths[idx]]
                output_lines = tokenizer.decode(tokens[input_lengths[idx]:])
                batch_output_lines.append(output_lines)
                batch_tokens.append(tokens[input_lengths[idx]:].cpu().tolist())
                input_lines = tokenizer.decode(tokens[:input_lengths[idx]])
                batch_input_lines.append(input_lines)

            samples.extend(batch_output_lines)

    out = {
        "prompt": prompt[0],
        "question": item["question"],
        "samples": samples,
        "gt_answer": item["answer"],
    }
    save_yaml(outpath, out)

    state_dict = {
        "wall_time": output_dict['wall_time'],
        "new_tokens": (output_lengths-input_lengths).cpu().tolist(),
        "inference_steps": output_dict['inference_steps'].cpu().tolist(),
        "output_text": batch_output_lines,
        "output_tokens": batch_tokens,
        "data_idx": item['id'],
        "input_lengths": input_lengths.cpu().tolist(),
        "output_lengths": output_lengths.cpu().tolist(),
    }
    # print(state_dict)
    json.dump(state_dict, log_file, ensure_ascii=False)
    log_file.write("\n")

def main():
    parser = argparse.ArgumentParser(description="GSM8K Speculative Decoding")
    
    # Model arguments
    parser.add_argument("--target_ckpt", required=True, help="Path to target model checkpoint")
    parser.add_argument("--draft_ckpt", required=True, help="Path to draft model checkpoint")
    parser.add_argument("--target_hf_model_name", required=True, help="HF model name for target config")
    parser.add_argument("--draft_hf_model_name", required=True, help="HF model name for draft config")
    parser.add_argument("--tensor_para_size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pipeline_para_size", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--int8_mode", type=int, default=0, help="Int8 mode")
    parser.add_argument("--lib_path", type=str, default="./lib/libth_transformer.so", help="Path to libth_transformer.so")
    parser.add_argument("--inference_data_type", type=str, default="fp16", help="Inference data type")
    parser.add_argument("--weights_data_type", type=str, default="fp16", help="Weights data type")
    
    # Data arguments
    parser.add_argument("--num_few_shot", type=int, default=2, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=-1, help="Maximum number of items to process")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset for dataset")
    parser.add_argument("--stride", type=int, default=1, help="Stride for dataset sampling")

    # Generation arguments
    parser.add_argument("--save_dir", required=True, help="Output directory for results")
    parser.add_argument("--draft_steps_max", type=int, default=5, help="Max draft tokens per step")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples per question")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Batch size for parallel generation")
    parser.add_argument("--output_len", type=int, default=128, help="Output sequence length")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling value")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling value")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")

    # method and name
    parser.add_argument("--method", type=str, default="SD_EMS", help="Method used for generation")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment")
    
    args = parser.parse_args()
    config = GSM8KConfig(args)
    
    # Prepare dataset
    test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))

    # add id to dataset
    for i, item in enumerate(train_dataset):
        item["id"] = i

    for i, item in enumerate(test_dataset):
        item["id"] = i
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        item["few_shot_items"] = few_shot_items

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")
    
    # Add few-shot examples
    random.seed(config.random_seed)
    
    # Apply dataset limits/offsets
    limit = config.limit if config.limit != -1 else len(test_dataset)
    test_dataset = test_dataset[config.offset:limit:config.stride]

    print(f"Total number of items to process: {len(test_dataset)}")

    # Load models and tokenizer
    target_gpt = load_opt_model(config.target_ckpt, config.target_config, config.tensor_para_size, config.lib_path, config.inference_data_type, config.int8_mode)
    draft_gpt = load_opt_model(config.draft_ckpt, config.draft_config, config.tensor_para_size, config.lib_path, config.inference_data_type, config.int8_mode)
    tokenizer = AutoTokenizer.from_pretrained(config.target_hf_model_name)
    
    # Create log file
    log_path = config.save_dir / "inference_logs.jsonl"
    with open(log_path, "w") as log_file:
        # Process items
        process_fn = partial(run_inference, 
                           config=config,
                           target_gpt=target_gpt,
                           draft_gpt=draft_gpt,
                           tokenizer=tokenizer,
                           log_file=log_file)
        
        for item in tqdm(test_dataset, desc="Processing GSM8K items"):
            start_time = datetime.now()
            process_fn(item)
            end_time = datetime.now()
            print(f"Time taken for item {item['id']}: {end_time - start_time}")

if __name__ == "__main__":
    main()