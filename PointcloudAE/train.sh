#!/bin/bash
LOSS=swd
F_TYPE=exp
G_TYPE=independent
INTER_DIM=64
PROJ_DIM=64
L=100
LR=0.001
KAPPA=1
SEED=1
GPU=0

lif [[ $LOSS == *"vsw"* ]]
then
    LOG_DIR=${LOSS}_kappa${KAPPA}_L${L}
elif [[ $LOSS == *"swd"* ]]
then
    LOG_DIR=${LOSS}_L${L}
else
    LOG_DIR=${LOSS}
fi

LOG_DIR=${LOG_DIR}_LR${LR}_seed${SEED}

echo $LOG_DIR

CUDA_VISIBLE_DEVICES=${GPU} python3 train.py --config="config.json" --logdir=logs/${LOG_DIR} --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" --loss=${LOSS} --autoencoder="pointnet" --f_type=${F_TYPE} --gradient_type=${G_TYPE}  --inter_dim=$INTER_DIM  --proj_dim=$PROJ_DIM --kappa=$KAPPA --num_projs=$L --learning_rate=${LR} --seed=${SEED}
