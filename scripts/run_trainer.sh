#!/bin/sh

python3 trainer.py \
--log_every 200 \
--batch_size 8 \
--max_epoch 100 \
--gamma1 4.0 \
--gamma2 5.0 \
--gamma2 10.0 \
--gamma2 5.0 \
--smooth_lambda 5.0