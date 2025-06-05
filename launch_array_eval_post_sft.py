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
task_name_save = "entity_sft-training"
model_name_ogs = ["Llama-3.1-8B-Instruct"]
# steer_types = ["negative-addition"]
# steer_types = ["positive-negative-addition-opposite"]
steer_types = ["positive-negative-addition-same"]

return_type = "prompt"
entity_types = ["song"]
known_unknown_split = 3
# entity_types = ["all"]
# entity_types = ["player", "city", "movie", "song", "all"]
# steer_poses = ["last", "entity"]  # "entity"



# Run the SLURM commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:

       futures = []
       for model_name_og in model_name_ogs:
                for entity_type in entity_types:

                    model_path = f"{task_name}_{model_name_og}_sft_{entity_type}_{known_unknown_split}"
                    huggingface_path= "winnieyangwannan/" + model_path
                    job_name  = model_path
                    save_path= f"/home/winnieyangwn/Output/{task_name_save}/{return_type}/{model_name_og}/{entity_type}/{known_unknown_split}/"


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


