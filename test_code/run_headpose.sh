#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --mem=60gb
#SBATCH --time=3-00:00:00
#SBATCH -J 1_all 
#SBATCH --exclude=node[002]

module load cuda10.1/toolkit/10.1.243
module load shared tensorflow2-py36-cuda10.1-gcc/2.0.0
module load openmpi/cuda/64/3.1.4
module load opencv3-py36-cuda10.1-gcc/3.4.10 
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0  

echo "Launching $1 ..."
dd=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES="0"

# Parameters:
# 1 data directory 
# 2 output directory 
# 3 model directory


#python3   test_headpose.py \
#          /home/yzhao11/headpose_final/Data \
####          /home/yzhao11/headpose_final/Results/new_with239_train14 \
#          /home/yzhao11/headpose_final/Models/new_with239_train14.h5 \
#          > ./log/log_new_with239_train14.txt 2>&1

#python3   test_headpose.py \
#          /home/yzhao11/headpose_final/Data \
#          /home/yzhao11/headpose_final/Results/published\
#          /home/yzhao11/headpose_final/Models/published.h5 \
#          > ./log/log_published.txt 2>&1

python3   test_headpose.py \
          /u/erdos/csga/jzhou40/MRI_Project/ \
          /u/erdos/csga/jzhou40/MRI_Project/test/test_rot \
          /u/erdos/csga/jzhou40/MRI_Project/train/M1_5/noskip_gradient_multiSubs_rot/weights_single_best.h5 \
          >./log_test_rot.txt 2>&1

# python3   test_headpose.py \
#           /home/yzhao11/headpose_final/Data \
#           /home/yzhao11/headpose_final/Results/all21\
#           /home/yzhao11/headpose_final/Models/all21.h5 \
#           > ./log/all21.txt 2>&1
