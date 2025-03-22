#! /bin/bash
set -ex
cd build
target_model=2.7b
draft_model=125m
output_len=256
num_samples=2
max_batch_size=2
tensor_para=1
draft_steps_max=5
num_few_shot=2
temperature=0.8
top_p=0.98
repetition_penalty=1.2
data_type=fp16
method=Vanilla
CUDA_VISIBLE_DEVICES=0
save_root=../outputs/llmonk_gsm8k
mkdir -p $save_root

name=$target_model-method$method-num_samples$num_samples-output_len$output_len-draft_steps_max$draft_steps_max-num_few_shot$num_few_shot-bs$max_batch_size

python ../llmonk/generate/gsm_8k_sd.py \
    --target_ckpt ../models/huggingface-models/c-model/opt-$target_model-$data_type/ \
    --draft_ckpt ../models/huggingface-models/c-model/opt-$draft_model-$data_type/ \
    --target_hf_model_name ../models/huggingface-models/hf_model/opt-$target_model/ \
    --draft_hf_model_name ../models/huggingface-models/hf_model/opt-$draft_model/ \
    --save_dir $save_root \
    --num_samples $num_samples \
    --max_batch_size $max_batch_size \
    --output_len $output_len \
    --draft_steps_max $draft_steps_max \
    --tensor_para_size $tensor_para \
    --num_few_shot $num_few_shot \
    --temperature $temperature \
    --repetition_penalty $repetition_penalty \
    --top_p $top_p \
    --lib_path lib/libth_transformer.so \
    --inference_data_type $data_type \
    --weights_data_type $data_type \
    --name $name \
    --method $method