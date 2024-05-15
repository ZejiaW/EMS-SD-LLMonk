import json
import numpy as np

warm_up = 0
num_samples = 480
output_len = 128
parallel = 8
for repeat in range(3):
    model = '6.7b'
    for bs in [1,2,4,8,12,16,20,24]:
        jsonl_files = [
            f"./outputs/{num_samples}-{parallel}/{model}-{num_samples}-output_len{output_len}-baseline-bs{bs}-repeat{repeat}.jsonl"
        ]
        for check_length in [2]:
            for copy_length in [7]:
                llma_methods = ["LLMA_vanilla", "LLMA_EMS_ablation_unpad_inputs", "LLMA_EMS_ablation_unpad_kv_cache","LLMA_EMS"] if bs != 1 else ["LLMA_vanilla"]
                for method in llma_methods:
                    jsonl_files.append(f"./outputs/{num_samples}-{parallel}/{model}-{num_samples}-output_len{output_len}-{method}-bs{bs}-check_length{check_length}-copy_length{copy_length}-repeat{repeat}.jsonl")
        for draft_steps_max in [4]:
            for method in ["SD_vanilla", "SD_EMS"] if bs != 1 else ["SD_vanilla"]:
                jsonl_files.append(f"./outputs/{num_samples}-{parallel}/{model}-{num_samples}-output_len{output_len}-{method}-bs{bs}-draft_steps_max{draft_steps_max}-repeat{repeat}.jsonl")
        
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as file:
                tokens_per_second = []
                tokens_per_second_v2 = []
                tokens_per_second_v3 = []
                tokens_all = []
                wall_time_all = []
                new_tokens_list = []
                compress_list = []
                inference_steps_list = []
                part_time_mean_list = []
                part_time_sum_list = []
                padding_ratio_list = []
                for idx, line in enumerate(file):
                    if idx >= warm_up:
                        json_obj = json.loads(line)
                        pad_tokens_ratio = (np.array(json_obj['pad_tokens'])/(np.array(json_obj['new_tokens']))).tolist()
                        padding_ratio_list.extend(pad_tokens_ratio)

                        # wall_time = np.array(json_obj['wall_time']).sum()/1000
                        # print(np.array(json_obj['wall_time'])[3:].reshape(-1,3).mean(axis=0))
                        # print(idx, np.array(json_obj['wall_time']).shape, ) # reshape(-1,3)
                        # print([json_obj['wall_time'][3*i:3*(i+1)] for i in range(len(json_obj['wall_time'])//3)])
                        wall_time = np.array(json_obj['wall_time'])[3:].reshape(-1,3).sum(axis=0)[:3].sum()/1000 #+ np.array(json_obj['wall_time'])[:3].sum()/1000
                        new_tokens = np.array(json_obj['new_tokens']).sum()
                        inference_steps = np.array(json_obj['inference_steps']).sum()
                        if wall_time == 0:
                            continue
                        tokens_per_second.append(new_tokens/wall_time)
                        tokens_per_second_v2.append([new_tokens, wall_time])
                        tokens_per_second_v3.extend([json_obj['new_tokens'][i]/(np.array(json_obj['wall_time'])[3:].reshape(-1,3).sum(-1)[:json_obj['inference_steps'][i]-1].sum()/1000) for i in range(len(json_obj['new_tokens']))])
                        new_tokens_list.append(np.array(json_obj['new_tokens']).mean())
                        tokens_all.append(new_tokens)
                        wall_time_all.append(wall_time)
                        compress_list.extend((np.array(json_obj['new_tokens'])/np.array(json_obj['inference_steps'])).tolist())
                        inference_steps_list.append(np.array(json_obj['inference_steps']).max())
                        part_time_mean_list.append([np.array(json_obj['wall_time'])[:3].sum()]+np.array(json_obj['wall_time'])[3:].reshape(-1,3).mean(axis=0).tolist())
                        part_time_sum_list.append([np.array(json_obj['wall_time'])[:3].sum()]+np.array(json_obj['wall_time'])[3:].reshape(-1,3).sum(axis=0).tolist())
                # print(",".join(["{:.2f}".format(t) for t in np.array(part_time_mean_list).mean(axis=0)]), np.array(part_time_mean_list).mean(axis=0)[3]/np.array(part_time_mean_list).mean(axis=0).sum())
                # print(",".join(["{:.2f}".format(t) for t in np.array(part_time_sum_list).mean(axis=0)]), np.array(part_time_sum_list).mean(axis=0)/np.array(part_time_sum_list).mean(axis=0).sum())
                print("{:.2f}".format(np.array(tokens_per_second).mean()), end=",")
                # print("{:.2f}".format(np.array(tokens_per_second_v2)[:,0].mean()/np.array(tokens_per_second_v2)[:,1].mean()), end=",")
                # print("{:.2f}".format(np.array(tokens_per_second_v3).mean()), end=",")
                # print("{:.2f}".format(np.array(padding_ratio_list).mean()), end=",")
                # print("{:.2f}".format(np.array(new_tokens_list).mean()), end=",")
                # print("{:.2f}".format(np.median(np.array(new_tokens_list))), end=",")
                # print("{:.2f}".format(np.percentile(np.array(new_tokens_list), 5)), end=",")
                # print("{:.2f}".format(np.array(compress_list).mean()), end=",")
                # print(np.array(compress_list).mean())

                # print("{:.2f}".format(np.array(inference_steps_list).mean()), end=",")
        print()