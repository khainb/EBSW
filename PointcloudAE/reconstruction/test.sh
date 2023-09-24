#!/bin/bash
LOG_DIR=logs/chamfer
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python3 reconstruction/reconstruction_test.py --config="reconstruction/config.json" --logdir=${LOG_DIR} --data_path="dataset/modelnet40_ply_hdf5_2048/"
