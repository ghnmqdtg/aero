#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

INPUT_SR_IN_K=4
OUTPUT_SR_IN_K=16
INPUT_SR=$(($INPUT_SR_IN_K * 1000))
OUTPUT_SR=$(($OUTPUT_SR_IN_K * 1000))

EXP_NAME=aero_${INPUT_SR_IN_K}-${OUTPUT_SR_IN_K}_512_64
OUTPUT_FOLDER=outputs/${INPUT_SR}/${OUTPUT_SR}

python test.py \
  dset=4-16 \
  experiment=${EXP_NAME}