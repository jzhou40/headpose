#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --mem=60gb
#SBATCH --time=3-00:00:00
#SBATCH -J test 
#SBATCH --exclude=node[001,002]

module load cuda10.1/toolkit/10.1.243
module load shared tensorflow2-py36-cuda10.1-gcc/2.0.0
module load openmpi/cuda/64/3.1.4
module load opencv3-py36-cuda10.1-gcc/3.4.10 
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0  

echo "Launching $1 ..."
dd=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES="0"

# Parameters:
# 1 rot/trans
# 2 data directory 
# 3 output directory 
# 4 model directory

python3   test_headpose.py \
		  trans \
          /u/erdos/csga/jzhou40/MRI_Project/ \
          /u/erdos/csga/jzhou40/MRI_Project/test_public_trans \
          /u/erdos/csga/jzhou40/MRI_Project/train/public_trans/noskip_gradient_multiSubs_trans/weights_single_best.h5 \
          >./log_test_public_trans.txt 2>&1
