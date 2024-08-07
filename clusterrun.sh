#!/bin/bash
#SBATCH -c 1 # Number of cores requested
#SBATCH -t 340 # Runtime in minutes
#SBATCH -p gpu_test # Partition to submit to
#SBATCH --gpus 1
#SBATCH --mem=16000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o /n/home09/dperrin/repos/MAE/logs/cluster/MAE_%j.out # Standard out goes to this file
#SBATCH -e /n/home09/dperrin/repos/MAE/logs/cluster/MAE_%j.err # Standard err goes to this filehostname
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

conda run -n MAE \
python3 mae_pretrain.py \
--data_path /n/holyscratch01/howe_lab_seas/dperrin/MAE-data/docker-data/fixedsize-torch/ \
--mask_ratio .75 \
--embeding_dim 96 \
--loging True \
--total_epoch 100 \
--warmup_epoch 20 \
--train_noise 0.0 \
--base_learning_rate 1.20e-2 \
--weight_decay .0005 \
--encoder_layer 1 \
--decoder_layer 1 
