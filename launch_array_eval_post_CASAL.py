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


gpus_per_node = 1

current_directory = os.getcwd()
print("current_directory:", current_directory)

task_name_eval = "mmlu"
task_name = "entity"
task_name_save = "entity_training"
model_name_ogs = ["Llama-3.1-8B-Instruct"]
# model_name_ogs = ["OLMoE-1B-7B-0924-Instruct"]
# model_name_ogs = ["Qwen2.5-VL-3B", "
# steer_types = ["negative-addition"]
# steer_types = ["positive-negative-addition-opposite"]
# steer_types = ["positive-negative-addition-same"]
steer_types = ["pnas"]

return_type = "prompt"
train_module =  "mlp" #"mlp" #block
steer_poses = ["last"]  # "entity"
steering_strength =4
entity_types = ["song"]
# entity_types = ["all"]
# entity_types = ["player", "city", "movie", "song", "all"]
# steer_poses = ["last", "entity"]  # "entity"
known_unknown_split = "3"
epoch = 19
lr = 1e-4


# Run the SLURM commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:

    futures = []
    for steer_pos in steer_poses:
        for model_name_og in model_name_ogs:
            for steer_type in steer_types:
                for entity_type in entity_types:
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
                        layers = [30]
                        # layers = [22]
                    elif "Qwen3-30B-A3B" in model_name_og:
                        layers = np.arange(0, 47, 2)
                    elif  "OLMoE-1B-7B-0924-Instruct" in model_name_og:
                        # layers = np.arange(0, 16, 2)    
                        layers = [10]
                    for layer in layers:
                        model_path = f"{task_name}_{model_name_og}_{train_module}_{steer_type}_layer_{layer}_{steering_strength}_{entity_type}_{known_unknown_split}_{lr}_{epoch}"
                        huggingface_path= "winnieyangwannan/" + model_path
                        job_name  = f"mmlu_eval_CASAL_{model_path}"
                        save_path= f"/fsx-project/winnieyangwn/Output/{task_name_save}/{return_type}/{model_name_og}/{steer_type}/{steer_pos}/layer_{layer}/strength_{steering_strength}/{entity_type}/{known_unknown_split}/{train_module}/epoch_{epoch}/lr_{lr}/{task_name_eval}"


                        slurm_cmd = f'''sbatch --account=genai_interns --qos=lowest \
                            --job-name={job_name} --nodes=1 --gpus-per-node={gpus_per_node} \
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


