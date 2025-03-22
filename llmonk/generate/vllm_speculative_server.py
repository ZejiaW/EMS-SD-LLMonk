from fastapi import FastAPI, Request
import uvicorn
import torch
import argparse
from typing import Optional
from pathlib import Path
import sys
import os

# Add custom GPT decoder path
sys.path.append(str(Path(__file__).parent.parent.parent / "examples/pytorch/gpt/utils"))
from gpt_decoder import Gpt
from gpt_specutive_decoding_EMS import gpt_speculative_decoding

app = FastAPI()
target_gpt = None
draft_gpt = None

class SpeculativeParams:
    def __init__(self, 
                 target_ckpt: str,
                 draft_ckpt: str,
                 target_config: dict,
                 draft_config: dict,
                 num_speculative_tokens: int = 5,
                 **kwargs):
        self.target_ckpt = target_ckpt
        self.draft_ckpt = draft_ckpt
        self.target_config = target_config
        self.draft_config = draft_config
        self.num_speculative_tokens = num_speculative_tokens
        self.__dict__.update(kwargs)

def load_opt_model(ckpt_path: str, config: dict, tensor_para_size: int, data_type: str):
    """Load OPT C-model"""
    return Gpt(
        num_heads=config['num_attention_heads'],
        size_per_head=config['hidden_size'] // config['num_attention_heads'],
        num_layers=config['num_hidden_layers'],
        vocab_size=config['vocab_size'],
        start_id=config['bos_token_id'],
        end_id=config['eos_token_id'],
        tensor_para_size=tensor_para_size,
        pipeline_para_size=1,  # Assuming no pipeline parallelism
        lib_path="./lib/libth_transformer.so",
        max_seq_len=config['max_position_embeddings'],
        layernorm_eps=1e-5,
        layernorm_type='pre_layernorm' if config['do_layer_norm_before'] else 'post_layernorm',
        activation_type='Relu' if config['activation_function'] == 'relu' else 'Gelu',
        has_post_decoder_layernorm=config['do_layer_norm_before'],
        int8_mode=0,
        inference_data_type=data_type,
        weights_data_type=data_type
    ).load(ckpt_path, data_type)

@app.post("/generate")
async def generate(request: Request):
    global target_gpt, draft_gpt
    request_dict = await request.json()
    
    # Convert request to OPT model parameters
    params = SpeculativeParams(
        target_ckpt=request_dict["target_ckpt"],
        draft_ckpt=request_dict["draft_ckpt"],
        target_config=request_dict["target_config"],
        draft_config=request_dict["draft_config"],
        num_speculative_tokens=request_dict.get("num_speculative_tokens", 5),
        max_tokens=request_dict.get("max_tokens", 50),
        temperature=request_dict.get("temperature", 0.7),
        top_p=request_dict.get("top_p", 0.9),
        stop=request_dict.get("stop", [])
    )
    
    # Run speculative decoding
    output = speculative_generate(
        request_dict["prompt"],
        params
    )
    
    return {"text": output}

def speculative_generate(prompt: str, params: SpeculativeParams):
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").int()
    input_length = torch.tensor([input_ids.size(1)], dtype=torch.int32)
    
    # Run draft model
    draft_output = draft_gpt.generate(
        input_token_ids=input_ids,
        input_lengths=input_length,
        gen_length=params.num_speculative_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=params.temperature,
        top_p=params.top_p
    )
    
    # Run target model verification
    target_output = target_gpt.generate(
        input_token_ids=input_ids,
        input_lengths=input_length,
        gen_length=params.num_speculative_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=params.temperature,
        top_p=params.top_p,
        speculative_tokens=draft_output['output_token_ids'][:, :, input_length[0]:]
    )
    
    # Decode final output
    return tokenizer.decode(target_output['output_token_ids'][0][0][input_length[0]:])

def initialize_models(target_ckpt: str, draft_ckpt: str,
                     target_config: dict, draft_config: dict,
                     tensor_para_size: int, data_type: str):
    global target_gpt, draft_gpt
    
    # Load target model
    target_gpt = load_opt_model(
        target_ckpt,
        target_config,
        tensor_para_size,
        data_type
    )
    
    # Load draft model
    draft_gpt = load_opt_model(
        draft_ckpt,
        draft_config,
        tensor_para_size,
        data_type
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ckpt", type=str, required=True)
    parser.add_argument("--draft-ckpt", type=str, required=True)
    parser.add_argument("--target-config", type=str, required=True)
    parser.add_argument("--draft-config", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-para-size", type=int, default=1)
    
    args = parser.parse_args()
    
    # Load configs
    def load_config(path):
        with open(path) as f:
            return json.load(f)
    
    initialize_models(
        args.target_ckpt,
        args.draft_ckpt,
        load_config(args.target_config),
        load_config(args.draft_config),
        args.tensor_para_size
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)