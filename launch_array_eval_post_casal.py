import os
import concurrent.futures
import subprocess
import re
from os.path import join
import argparse
import numpy as np
# model_path = "meta-llama/Llama-3.1-8B"
def run_subprocess_slurm(command):
    # Execute the sbatch command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output and error, if any
    print("command:", command)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Get the job ID for linking dependencies
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
    else:
        job_id = None

    return job_id

account = "genai_interns"
nodes = 1  # 1 node is enough for most models
gpus_per_node= 1
qos ="lowest" # "genai_interns"  # or "lowest"

task_name_eval = "mmlu"
algorithm = "casal"
# model_name_ogs = ["gemma-2-9b-it"]
# model_name_ogs = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
model_name_ogs = ["Llama-3.1-8B-Instruct"]

task_name_train = "entity"
steer_types = ["pnas"]
known_unknown_split = "37"
train_module = "mlp" #  "mlp-up" # "experts" #  #  "experts" #     #block
steering_strengths = [4]
entity_type = "all"
epochs = [15, 20]
batch_size = 64 # 64
n_train =1280  # 1280 # 32 # 
n_val = 1280 # 1280 # 32 #
n_test = 1280 #  # 1280 # 32 #
max_new_tokens =100 #  100 # 100 # 100 # 100
# lrs = [1e-5, 5e-5 ,1e-4, 5e-4, 1e-3, 5e-3]
lr =1e-4

# task_name = "popqa"


# task_name = "entity-prefix"


# Run the SLURM commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:

    futures = []
    for epoch in epochs:
        for model_name_og in model_name_ogs:
            for steer_type in steer_types:
                for steering_strength in steering_strengths:
                    if  "Qwen2.5-VL-3B" in model_name_og:
                        layers = np.arange(0, 35, 2)
                        layers = [15]
                    elif  "Qwen2.5-VL-7B" in model_name_og:
                        layers = np.arange(0, 27, 2)
                    elif  "gemma-2-9b-it" in model_name_og:
                        layers = np.arange(0, 42, 2)
                        # layers = [40]
                    elif "Llama-3.1-8B-Instruct" in model_name_og:
                        layers = np.arange(0, 32+2, 2)
                        layers = [22]
                        # layers = [22]
                    elif "Qwen3-30B-A3B" in model_name_og:
                        layers = np.arange(0, 47, 2)
                    elif  "OLMoE-1B-7B-0924-Instruct" in model_name_og:
                        # layers = np.arange(0, 16, 2)    
                        layers = [10]
                    for layer in layers:
                        model_path = f"{task_name_train}_{model_name_og}_{train_module}_{steer_type}_layer_{layer}_{steering_strength}_{entity_type}_{known_unknown_split}_{lr}_{n_train}_{epoch}"
                        huggingface_path= "winnieyangwannan/" + model_path
                        job_name  = f"general_eval_post_{algorithm}_{model_path}"
                        save_path= f"/fsx-project/winnieyangwn/Output/{task_name_train}_training/prompt/{model_name_og}/{steer_type}/last/layer_{layer}/strength_{steering_strength}/{entity_type}/{known_unknown_split}/{train_module}/epoch_{epoch}/lr_{lr}/n_train{n_train}/{task_name_eval}"

                        slurm_cmd = f'''sbatch --account={account} --qos={qos} \
                            --job-name={job_name} --nodes={nodes} --gpus-per-node={gpus_per_node} \
                            --time=24:00:00 --output=LOGS/{job_name}.log \
                            --wrap="\
                                lm_eval --model hf \
                                --model_args pretrained={huggingface_path} \
                                --tasks {task_name_eval} \
                                --log_samples \
                                --output_path {save_path}; \
                        "'''
                        job_id = run_subprocess_slurm(slurm_cmd)
                        print("job_id:", job_id)


