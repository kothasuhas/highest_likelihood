import os
import subprocess

models = ['EleutherAI/pythia-70m']
desired_lengths = [9]


sbatch_base = "#!/bin/bash" \
+ "\n#SBATCH --job-name=exec" \
+ "\n#SBATCH --output=/data/suhas_kotha/slurmjobs/%j.out" \
+ "\n#SBATCH --cpus-per-task=1" \
+ "\n#SBATCH --gpus-per-node=1" \
+ "\n#SBATCH --tasks-per-node=1" \
+ "\n#SBATCH --mem=0G" \
+ "\n#SBATCH --time=12:00:00" \
+ "\n#SBATCH --partition=compute" 

sbatch_base += "\n cd /data/suhas_kotha/playground/highest_likelihood ; "
sbatch_base += " source activate trl ; "

for model in models:
    sbatch_base_temp = sbatch_base
    for length in desired_lengths:
        print(f"Model: {model}, Length: {length}")
        command = f" python3 highest_likelihood.py --model_name {model} --num_tokens 188 --desired_length {length} --greedy ; "
        sbatch_base_temp += command

    f = open("/data/suhas_kotha/template.sh", "w")
    f.write(sbatch_base_temp)
    f.close()

    subprocess.run(f'sbatch /data/suhas_kotha/template.sh', shell=True)
