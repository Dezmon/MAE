#!/bin/bash
python3 mae_pretrain.py \
--data_path docker-data/fixedsize-torch/ \
--embeding_dim 36 \
--loging True \
--total_epoch 1000 \
--warmup_epoch 200 \
--train_noise .01 \
--base_learning_rate 1.20e-4 \
--encoder_layer 1 \
--decoder_layer 1 

pkill wandb