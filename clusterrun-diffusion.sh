#!/bin/bash
#SBATCH -c 1 # Number of cores requested
#SBATCH -t 340 # Runtime in minutes
#SBATCH -p gpu_test # Partition to submit to
#SBATCH --gpus 1
#SBATCH --mem=16000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o /n/home09/dperrin/repos/MAE/logs/cluster/DIFF_%j.out # Standard out goes to this file
#SBATCH -e /n/home09/dperrin/repos/MAE/logs/cluster/DIFF_%j.err # Standard err goes to this filehostname
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

conda run -n MAE \
python3 denoising_diffusion.py\

