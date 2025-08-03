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

qos = "genai_interns"
gpus_per_node = 1
nodes = 2


current_directory = os.getcwd()
print("current_directory:", current_directory)

task_name_eval = "mmlu"
task_name_train = "entity"
model_paths = ["meta-llama/Llama-3.1-8B-Instruct"]
# model_paths = ["allenai/OLMoE-1B-7B-0924-Instruct"]

# steer_types = ["negative-addition"]
# steer_types = ["positive-negative-addition-opposite"]
# entity_types = ["all"]
# entity_types = ["player", "city", "movie", "song", "all"]
# steer_poses = ["last", "entity"]  # "entity"

known_unknown_split = "37"

# Run the SLURM commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:

       futures = []
       for model_path in model_paths:

                    huggingface_path= model_path
                    model_name = os.path.basename(model_path)
                    job_name  = f"mmlu_eval_baseline_{model_name}"
                    save_path = f"/fsx-project/winnieyangwn/Output/{task_name_train}_baseline/{model_name}/all/{known_unknown_split}/{task_name_eval}"

                    slurm_cmd = f'''sbatch --account=genai_interns --qos={qos} \
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


