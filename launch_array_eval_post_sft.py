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
algorithm = "sft"
# model_name_ogs = ["gemma-2-9b-it"]
# model_name_ogs = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
model_name_ogs = ["Llama-3.1-8B-Instruct"]

task_name_train = "entity"
steer_types = ["pnas"]
known_unknown_split = "37"
train_module = "mlp" #  "mlp-up" # "experts" #  #  "experts" #     #block
entity_type = "all"
epochs = [10, 15, 20]
batch_size = 64 # 64
n_train =1280  # 1280 # 32 # 
n_val = 1280 # 1280 # 32 #
n_test = 1280 #  # 1280 # 32 #
max_new_tokens =100 #  100 # 100 # 100 # 100
# lrs = [1e-5, 5e-5 ,1e-4, 5e-4, 1e-3, 5e-3]
lr =1e-4
lora = True
# task_name = "popqa"


# task_name = "entity-prefix"


# Run the SLURM commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:

    futures = []
    for epoch in epochs:
        for model_name_og in model_name_ogs:
                
                        model_path = f"{task_name_train}_sft_{model_name_og}_lora_{lora}_lr_{lr}_{n_train}_{entity_type}_{known_unknown_split}_epoch_{epoch}"
                        huggingface_path= "winnieyangwannan/" + model_path
                        job_name  = f"general_eval_post_{algorithm}_{model_path}"
                        save_path= f"/fsx-project/winnieyangwn/Output/{task_name_train}_{algorithm}-training/prompt/{model_name_og}/lora_{lora}/lr_{lr}/{n_train}/{entity_type}/{known_unknown_split}/epoch_{epoch}/{task_name_eval}"

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


