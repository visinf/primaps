#!/usr/bin/env bash


name='reproduce_main_results'

# Precompute PriMaPs pseudo labels 
python train.py --precomp-primaps --threshold 0.4 --dataset-root /path/to/dataset --backbone-arch dino_vits --backbone-patch 8 --dino-block 1 --batch-size 32 --validation-resize 320 --crop-size 320 --num-workers 8 --augs-randcrop-scale 1. 1. --augs-randcrop-ratio 1. 1. --log-name $name --gpu-ids 0

# Get initialization checkpoint
python train.py --pcainit --dataset-root /path/to/dataset --backbone-arch dino_vits --backbone-patch 8 --dino-block 1 --batch-size 32 --train-state baseline --validation-resize 320 --crop-size 320 --num-workers 4 --num-epochs 2 --linear-lr 5e-3 --augs-randcrop-scale 1. 1. --augs-randcrop-ratio 1. 1. --log-name $name --gpu-ids 0

# Optimize class prototypes with PriMaPs pseudo labels
primaps='/path/to/pseudo/labels/' 
init_checkpoint='/path/to/checkpoint/last.ckpt'

python train.py --log-name $name --cluster-ckpt-path $init_checkpoint --student-augs --dataset-root /path/to/dataset --backbone-arch dino_vits --backbone-patch 8 --dino-block 1 --batch-size 32 --validation-resize 320 --crop-size 320 --num-workers 4 --num-epochs 50 --linear-lr 5e-3 --ema-update-step 10 --ema-decay 0.98 --precomp-primaps-root $primaps --seghead-lr 5e-3 --seghead-arch 'linear' --augs-randcrop-scale 1. 1. --augs-randcrop-ratio 1. 1. --gpu-ids 0