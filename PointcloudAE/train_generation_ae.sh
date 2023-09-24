#!/bin/bash
LOSS=emd
F_TYPE=lin_attn
INTER_DIM=64
PROJ_DIM=64
L=100
KAPPA=1
SEED=1
GPU=0

if [[ $LOSS == *"amortized"* ]]
then
    LOG_DIR=${LOSS}_${F_TYPE}
    if [[ $F_TYPE == *'attn'* ]]
    then
        LOG_DIR=${LOG_DIR}_dim${INTER_DIM}
        if [[ $F_TYPE == *'lin_attn'* ]]
        then
            LOG_DIR=${LOG_DIR}_proj${PROJ_DIM}
        fi
    fi

    if [[ $LOSS == *"vsw"* ]]
    then
        LOG_DIR=${LOG_DIR}_kappa${KAPPA}_L${L}
    fi
elif [[ $LOSS == *"vsw"* ]]
then
    LOG_DIR=${LOSS}_kappa${KAPPA}_L${L}
elif [[ $LOSS == *"swd"* ]]
then
    LOG_DIR=${LOSS}_L${L}
else
    LOG_DIR=${LOSS}
fi

LOG_DIR=${LOG_DIR}_seed${SEED}

echo $LOG_DIR

CUDA_VISIBLE_DEVICES=${GPU} python3 train.py --config="generation_ae_config.json" --logdir=generation_logs/${LOG_DIR} --data_path="dataset/shapenet_chair/train.npz" --loss=${LOSS} --autoencoder="pointnet" --f_type=${F_TYPE}  --inter_dim=$INTER_DIM  --proj_dim=$PROJ_DIM --kappa=$KAPPA --num_projs=$L --seed=${SEED}
