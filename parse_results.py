import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
def load_logs(log_file_path):
    """Load the JSON lines log file into a pandas DataFrame."""
    data = []
    with open(log_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def process_wall_times(df):
    """Process wall times into separate columns for each period."""
    # Initialize lists to store each period's times
    context_decoder_times = []
    draft_decoder_times = []
    main_decoder_times = []
    post_process_times = []

    # Iterate over each sample's wall time list
    for wall_time_list in df['wall_time']:
        # Ensure the wall_time list has the expected number of periods
        if len(wall_time_list) >= 4:
            context_decoder_times.append(wall_time_list[0])
            draft_decoder_times.append(sum(wall_time_list[1::3]))
            main_decoder_times.append(sum(wall_time_list[2::3]))
            post_process_times.append(sum(wall_time_list[3::3]))
        else:
            # Handle cases where the wall_time list is shorter than expected
            context_decoder_times.append(float('nan'))
            draft_decoder_times.append(float('nan'))
            main_decoder_times.append(float('nan'))
            post_process_times.append(float('nan'))

    # Add the times to the DataFrame
    df['context_decoder_time'] = context_decoder_times
    df['draft_decoder_time'] = draft_decoder_times
    df['main_decoder_time'] = main_decoder_times
    df['post_process_time'] = post_process_times

def plot_average_wall_time(df, log_file_path):
    """Plot the average wall time for each period across all samples."""
    # Calculate average wall time for each period
    avg_wall_time = df[['context_decoder_time', 'draft_decoder_time', 'main_decoder_time', 'post_process_time']].mean()

    # Define labels for each period
    labels = ['Context Decoder', 'Draft Decoder', 'Main Decoder', 'Post-Processing']

    print("Average wall time: ", avg_wall_time.mean())

    # Plot the average wall times
    plt.figure(figsize=(12, 8))
    sns.barplot(x=labels, y=avg_wall_time)
    # plt.title('Average Wall Time for Each Period', fontsize=16)
    plt.xlabel('Period', fontsize=20)
    plt.ylabel('Average Time (ms)', fontsize=20)
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 2000)
    plt.show()
    plt.savefig(f"{log_file_path.parent}/average_wall_time.png")

def process_wall_times_vanilla(df):
    """Process wall times into separate columns for each period."""
    # Initialize lists to store each period's times
    context_decoder_times = []
    decoder_times = []
    post_process_times = []

    # Iterate over each sample's wall time list
    for wall_time_list in df['wall_time']:
        # Ensure the wall_time list has the expected number of periods
        if len(wall_time_list) >= 2:
            context_decoder_times.append(wall_time_list[0])  # Context Decoder
            # Decoder and post-process times are interleaved after the first element
            decoder_times.append(sum(wall_time_list[1::2]))  # Sum of all decoder times
            post_process_times.append(sum(wall_time_list[2::2]))  # Sum of all post-process times
        else:
            # Handle cases where the wall_time list is shorter than expected
            context_decoder_times.append(float('nan'))
            decoder_times.append(float('nan'))
            post_process_times.append(float('nan'))

    # Add the times to the DataFrame
    df['context_decoder_time'] = context_decoder_times
    df['decoder_time'] = decoder_times
    df['post_process_time'] = post_process_times

def plot_average_wall_time_vanilla(df, log_file_path):
    """Plot the average wall time for each period across all samples."""
    # Calculate average wall time for each period
    avg_wall_time = df[['context_decoder_time', 'decoder_time', 'post_process_time']].mean()
    
    # Define labels for each period
    labels = ['Context Decoder', 'Decoder', 'Post-Processing']

    print("Average wall time: ", avg_wall_time.mean())

    # Plot the average wall times
    plt.figure(figsize=(12, 8))
    sns.barplot(x=labels, y=avg_wall_time)
    # plt.title('Average Wall Time for Each Inference Period', fontsize=16)
    plt.xlabel('Period', fontsize=20)
    plt.ylabel('Average Time (ms)', fontsize=20)
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.ylim(0, 2000)
    plt.show()
    plt.savefig(f"{log_file_path.parent}/average_wall_time_vanilla.png")

def plot_inference_steps(df, log_file_path):
    """Plot the number of inference steps for each item."""
    print("Average inference steps: ", df['inference_steps'].apply(sum).mean())
    plt.figure(figsize=(12, 6))
    sns.histplot(df['inference_steps'].apply(sum), bins=30, kde=True)
    # plt.title('Distribution of Total Inference Steps', fontsize=16)
    plt.xlabel('Total Inference Steps', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.savefig(f"{log_file_path.parent}/inference_steps_distribution.png")

def plot_new_tokens(df, log_file_path):
    """Plot the number of new tokens generated for each item."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['new_tokens'].apply(sum), bins=30, kde=True)
    # plt.title('Distribution of New Tokens Generated', fontsize=16)
    plt.xlabel('New Tokens', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.savefig(f"{log_file_path.parent}/new_tokens_distribution.png")

def main(log_file_path, is_vanilla=False):
    # Load the logs into a DataFrame
    df = load_logs(log_file_path)
    # print(df.head())
    # Plot the wall time for each inference step
    if is_vanilla:
        process_wall_times_vanilla(df)
        plot_average_wall_time_vanilla(df, log_file_path)
    else:
        process_wall_times(df)
        plot_average_wall_time(df, log_file_path)

    # Plot the number of inference steps for each item
    plot_inference_steps(df, log_file_path)

    # Plot the number of new tokens generated for each item
    plot_new_tokens(df, log_file_path)

if __name__ == "__main__":
    # Path to the inference logs JSONL file
    root_dir = "/root/data_new/zejia/workspace/psl/EMS-SD-LLMonk/outputs/llmonk_gsm8k"
    # experiment_name = "2.7b-methodVanilla-num_samples2-output_len256-draft_steps_max5-num_few_shot2-bs2"
    experiment_name = "2.7b-methodSD_EMS-num_samples2-output_len256-draft_steps_max5-num_few_shot2-bs2"
    # experiment_name = "6.7b-methodSD_EMS-num_samples2-output_len256-draft_steps_max5-num_few_shot2-bs2"
    log_file_path = Path(f"{root_dir}/{experiment_name}/inference_logs.jsonl")
    main(log_file_path, is_vanilla="Vanilla" in experiment_name)