#!/bin/bash
#SBATCH -o ./slurm-%j.out
#SBATCH -e ./slurm-%j.err 
#SBATCH --partition=a100
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 32     # 8 MPI processes per node
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:1


__conda_setup="$('$HOME/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate torch_env


python ./predict_hform.py


