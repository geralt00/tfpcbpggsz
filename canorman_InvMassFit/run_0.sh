#!/bin/sh
echo Hello from $USER on $HOSTNAME
marco=/software/pc24403/miniconda3/envs/tfdev/bin/python3
script=/shared/scratch/pc24403/sim_fit/ana/tf_fit.py

GPU=5

$marco $script --index 501 --gpu $GPU 
