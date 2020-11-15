#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=180gb
#SBATCH --time=3-00:00:00
#SBATCH -J train
#SBATCH --mail-user=jzhou40@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=node[001,003]


module load cuda10.1/toolkit/10.1.243
module load shared tensorflow2-py36-cuda10.1-gcc/2.0.0
module load openmpi/cuda/64/3.1.4
module load opencv3-py36-cuda10.1-gcc/3.4.10
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0


echo "Launching $1 ..."
dd=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES="0"
# 1, train script; 2, skip or noskip ( intermediate frames) 3, original or image gradient
# 4, save results directory 5, rot or trans 6, batch size (eg: batch size = 10, means all 13 subs *10 = 130 per input)
# 7 epoch numbers 8 combined(gradient & original)  or single (original/gradient)

python3   main.py \
                noskip      \
                gradient    \
               /u/erdos/csga/jzhou40/MRI_Project/train/public_trans/  \
                trans     \
                5       \
                200     \
                combined    \
                >./log_public_trans.txt 2>&1

