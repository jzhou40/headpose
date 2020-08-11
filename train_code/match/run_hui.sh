#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=180gb
#SBATCH --time=3-00:00:00
#SBATCH -J M1_alltr
#SBATCH --mail-user=jzhou40@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=node[001,003]


module load cuda10.1/toolkit/10.1.243
module load shared tensorflow2-py36-cuda10.1-gcc/2.0.0
module load openmpi/cuda/64/3.1.4
module load opencv3-py36-cuda10.1-gcc/3.4.9
module load ml-pythondeps-py36-cuda10.1-gcc/3.2.3

echo "Launching $1 ..."
dd=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES="0"
#
#
# 1, train script; 2, skip or noskip ( intermediate frames) 3, original or image gradient
# 4, save results directory 5, rot or trans 6, batch size (eg: batch size = 10, means all 13 subs *10 = 130 per input)
# 7 epoch numbers 8 combined(gradient & original)  or single (original/gradient)

python3   main_hui.py \
                noskip      \
                gradient    \
               /u/erdos/csga/jzhou40/MRI_Project/train/M1_all_trans2/  \
                trans     \
                5       \
                200     \
                combined    \
                >./log_M1_all_trans2.txt 2>&1


# 1, test script; 2, skip or noskip ( intermediate frames) 3, original or image gradient
# 4 load data directory 5, save results directory 6, rot or trans 7, model location( must with ".h5")
# 8 r2_p_value: all --> with intermediate lables, part --> only labels every 1.3s

#train_comb_x+y_noskip_w_200true
#
#python3   test_regression_toCSV_copy.py \
#           noskip \
#           gradient \
#           /u/erdos/csga/jzhou40/MRI_Project/  \
#           /u/erdos/csga/jzhou40/MRI_Project/test/test_comb_5img_6len_rm_cov512*2+256+64_2val_unmatch \
#           rot \
#           /u/erdos/csga/jzhou40/MRI_Project/train/comb_5img_6len_rm_cov512*2+256+64_2val_unmatch/noskip_gradient_multiSubs_rot/weights_single_best.h5 \
#           full \
#           combined \
#           >../test/log_test_comb_5img_6len_rm_cov512*2+256+64_2val_unmatch.txt 2>&1

# python3   test_regression_toCSV_hui.py \
#           noskip \
#           gradient \
#           /u/erdos/csga/jzhou40/MRI_Project/  \
#           /u/erdos/csga/jzhou40/MRI_Project/test/M1_1_trans \
#           trans \
#           /u/erdos/csga/jzhou40/MRI_Project/train/M1_1_trans/noskip_gradient_multiSubs_rot/weights_single_best.h5 \
#           full \
#           combined \
#           >../test/log_M1_1_trans.txt 2>&1

