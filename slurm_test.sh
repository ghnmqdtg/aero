#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

INPUT_SR_IN_K=4
OUTPUT_SR_IN_K=16

EXP_NAME=aero_${INPUT_SR_IN_K}-${OUTPUT_SR_IN_K}_512_64

python test.py \
  dset=${INPUT_SR_IN_K}-${OUTPUT_SR_IN_K} \
  experiment=${EXP_NAME}