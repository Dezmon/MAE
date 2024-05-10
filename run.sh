#!/bin/bash
python3 mae_pretrain.py \
--data_path docker-data/fixedsize-torch/ \
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

pkill wandb