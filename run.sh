#!/bin/bash
python3 mae_pretrain.py \
--data_path docker-data/fixedsize-torch/ \
--mask_ratio .75 \
--embeding_dim 384 \
--loging True \
--total_epoch 1 \
--warmup_epoch 200 \
--train_noise 0.0 \
--base_learning_rate 1.20e-3 \
--weight_decay .05 \
--encoder_layer 1 \
--decoder_layer 1 \
--batch_size 1

pkill wandb