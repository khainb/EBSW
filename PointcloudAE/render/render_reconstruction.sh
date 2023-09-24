#!/bin/bash
FILE=reconstruct_random_50_shapenetcore55.npy

FOLDER=raw
COLOR=tab:gray
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}

FOLDER=swd
COLOR=tab:red
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}

FOLDER=msw_iter50
COLOR=tab:green
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}

FOLDER=vsw_kappa1_iter50
COLOR=tab:blue
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}

FOLDER=amortized_vsw_lin_attn_dim64_proj64_kappa1
COLOR=tab:purple
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}