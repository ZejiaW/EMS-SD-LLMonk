import os
import re
import yaml
from tqdm import tqdm
from glob import glob

def check_correctness(dataset_dir):
    total = 0
    correct = 0
    
    for filepath in tqdm(glob(os.path.join(dataset_dir, "*.yaml"))):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract ground truth answer using regex
        gt_match = re.search(r'####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)', data['gt_answer'])
        if not gt_match:
            continue  # skip invalid entries
        gt_number = gt_match.group(1).replace(',', '')  # Normalize numbers with commas
        
        # Create regex pattern to match exact number with word boundaries
        pattern = re.compile(rf'\b{re.escape(gt_number)}\b')
        
        # Check all samples
        found = False
        for sample in data['samples']:
            # Normalize sample by removing commas before checking
            clean_sample = sample.replace(',', '')
            if pattern.search(clean_sample):
                found = True
                break
        
        total += 1
        if found:
            correct += 1
    
    print(f"Accuracy: {correct}/{total} ({correct/total:.2%})")

# Usage example:
root_dir = "/root/data_new/zejia/workspace/psl/EMS-SD-LLMonk/outputs/llmonk_gsm8k"
# experiment_name = "2.7b-methodSD_EMS-num_samples2-output_len256-draft_steps_max5-num_few_shot2-bs2"
# experiment_name = "2.7b-methodVanilla-num_samples2-output_len256-draft_steps_max5-num_few_shot2-bs2"
experiment_name = "6.7b-methodSD_EMS-num_samples2-output_len256-draft_steps_max5-num_few_shot2-bs2"
check_correctness(f"{root_dir}/{experiment_name}/results")