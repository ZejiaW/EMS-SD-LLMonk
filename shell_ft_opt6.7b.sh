set -ex
cd build
num_samples=480
output_len=128
tensor_para=1
CUDA_VISIBLE_DEVICES=0
save_root=../outputs/$num_samples-$tensor_para
draft_model=125m
mkdir -p $save_root
mkdir -p $save_root/logs
for r in {0..2}
do
    for model in 2.7b
    do
        # bs=1
        
        # # baseline
        # method=baseline
        # name=$model-$num_samples-output_len$output_len-$method-bs$bs-repeat$r
        # mpirun -n $tensor_para python ../examples/pytorch/gpt/opt_summarization.py --hf_model_name ../models/huggingface-models/hf_model/opt-$model/ --ft_model_location ../models/huggingface-models/c-model/opt-$model-fp16/ --data_type fp16 --verbose --weights_data_type fp16 --use_gpt_decoder_ops --summarize --tensor_para_size $tensor_para --max_batch_size $bs --name $name --num_samples $num_samples --output_len $output_len --save_root $save_root > $save_root/logs/$name.log
        # # LLMA
        # for check_length in 2
        # do
        #     for copy_length in 7
        #     do
        #         for method in LLMA_vanilla
        #         do
        #             echo $method
        #             name=$model-$num_samples-output_len$output_len-$method-bs$bs-check_length$check_length-copy_length$copy_length-repeat$r
        #             mpirun -n $tensor_para python ../examples/pytorch/gpt/opt_summarization.py --hf_model_name ../models/huggingface-models/hf_model/opt-$model/ --ft_model_location ../models/huggingface-models/c-model/opt-$model-fp16/ --data_type fp16 --verbose --weights_data_type fp16 --use_gpt_decoder_ops --summarize --tensor_para_size $tensor_para --max_batch_size $bs --name $name --method $method --check_length $check_length --copy_length $copy_length --num_samples $num_samples --output_len $output_len --save_root $save_root > $save_root/logs/$name.log
        #         done
        #     done
        # done
        # # draft model predication
        # for draft_steps_max in 4
        # do
        #     for method in  SD_vanilla
        #     do
        #         name=$model-$num_samples-output_len$output_len-$method-bs$bs-draft_steps_max$draft_steps_max-repeat$r
        #         mpirun -n $tensor_para python ../examples/pytorch/gpt/opt_summarization.py --hf_model_name ../models/huggingface-models/hf_model/opt-$model/ --ft_model_location ../models/huggingface-models/c-model/opt-$model-fp16/ --data_type fp16 --verbose --weights_data_type fp16 --use_gpt_decoder_ops --summarize --tensor_para_size $tensor_para --max_batch_size $bs --name $name --method $method --num_samples $num_samples --output_len $output_len --draft_ft_model_location ../models/huggingface-models/c-model/opt-$draft_model-fp16/ --draft_hf_model_name ../models/huggingface-models/hf_model/opt-$draft_model/ --draft_steps_max $draft_steps_max --save_root $save_root > $save_root/logs/$name.log
        #     done
        # done
        
        # bs > 1
        for bs in 2 # 24 20 16 12 8 4 2
        do
            # baseline
            # method=baseline
            # name=$model-$num_samples-output_len$output_len-$method-bs$bs-repeat$r
            # mpirun -n $tensor_para python ../examples/pytorch/gpt/opt_summarization.py --hf_model_name ../models/huggingface-models/hf_model/opt-$model/ --ft_model_location ../models/huggingface-models/c-model/opt-$model-fp16/ --data_type fp16 --verbose --weights_data_type fp16 --use_gpt_decoder_ops --summarize --tensor_para_size $tensor_para --max_batch_size $bs --name $name --num_samples $num_samples --output_len $output_len --save_root $save_root > $save_root/logs/$name.log
            # # LLMA
            # for check_length in 2
            # do
            #     for copy_length in 7
            #     do
            #         for method in LLMA_EMS LLMA_vanilla LLMA_EMS_ablation_unpad_inputs LLMA_EMS_ablation_unpad_kv_cache
            #         do
            #             echo $method
            #             name=$model-$num_samples-output_len$output_len-$method-bs$bs-check_length$check_length-copy_length$copy_length-repeat$r
            #             mpirun -n $tensor_para python ../examples/pytorch/gpt/opt_summarization.py --hf_model_name ../models/huggingface-models/hf_model/opt-$model/ --ft_model_location ../models/huggingface-models/c-model/opt-$model-fp16/ --data_type fp16 --verbose --weights_data_type fp16 --use_gpt_decoder_ops --summarize --tensor_para_size $tensor_para --max_batch_size $bs --name $name --method $method --check_length $check_length --copy_length $copy_length --num_samples $num_samples --output_len $output_len --save_root $save_root > $save_root/logs/$name.log
            #         done
            #     done
            # done
            # draft model predication
            for draft_steps_max in 2 4 8 12 16 20 24
            do
                for method in  SD_EMS SD_vanilla
                do
                    name=$model-$num_samples-output_len$output_len-$method-bs$bs-draft_steps_max$draft_steps_max-repeat$r
                    mpirun -n $tensor_para python ../examples/pytorch/gpt/opt_summarization.py --hf_model_name ../models/huggingface-models/hf_model/opt-$model/ --ft_model_location ../models/huggingface-models/c-model/opt-$model-fp16/ --data_type fp16 --verbose --weights_data_type fp16 --use_gpt_decoder_ops --summarize --tensor_para_size $tensor_para --max_batch_size $bs --name $name --method $method --num_samples $num_samples --output_len $output_len --draft_ft_model_location ../models/huggingface-models/c-model/opt-$draft_model-fp16/ --draft_hf_model_name ../models/huggingface-models/hf_model/opt-$draft_model/ --draft_steps_max $draft_steps_max --save_root $save_root > $save_root/logs/$name.log
                done
            done
        done
    done
done
cd ..