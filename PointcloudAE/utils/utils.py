import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time
from typing import List

import torch
from tqdm import tqdm


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def initialize_main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config file")
    parser.add_argument("--logdir", help="folder to save results")
    parser.add_argument("--data_path", type=str, default=None, help="path to data")
    args = parser.parse_args()
    config = args.config
    logdir = args.logdir
    data_path = args.data_path
    print("Save results in: ", logdir)
    args = json.load(open(config))

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(">Logdir was created successfully at: ", logdir)
    else:
        print(">Logdir exists.")

    if "save_config" in kwargs.keys():
        saving_config_filename = kwargs["save_config"]
        print(">In saving config mode.")
        fname = os.path.join(logdir, saving_config_filename)
        with open(fname, "w") as fp:
            json.dump(args, fp, indent=4)
    else:
        print(">Not in saving config mode.")

    # print hyperparameters
    print("You have 5s to check the hyperparameters below.")
    print(args)
    time.sleep(5)
    if data_path is None:
        return args, logdir
    else:
        return args, logdir, data_path


def load_model_for_evaluation(model, model_path, **kwargs):
    device = next(model.parameters()).device
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        model.load_state_dict(torch.load(model_path, map_location=device)[kwargs["key"]])
    model.eval()
    return model


def create_save_folder(logdir, save_folder_branch):
    save_folder = osp.join(logdir, save_folder_branch)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(">Save_folder was created successfully at:", save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        print(">Save_folder was created successfully at:", save_folder)
        # print(">Save_folder {} is existing.".format(save_folder))
        # print(">Do you want to remove it?")
        # answer = None
        # while answer not in ("yes", "no"):
        #     try:
        #         answer = input("Enter 'yes' or 'no': ")
        #     except:
        #         print("Please enter your answer again.")
        #         answer = raw_input("Enter 'yes' or 'no': ")
        #     if answer == "yes":
        #         shutil.rmtree(save_folder)
        #         os.makedirs(save_folder)
        #     elif answer == "no":
        #         print("SOME FILES WILL BE OVERWRITTEN OR APPENDED.")
        #         print("If you do not want this, please stop during next 5s.")
        #         time.sleep(5)
        #     else:
        #         print("Please enter yes or no.")
    return {"save_folder": save_folder}


def evaluate_on_dataset(evaluator, model, dataloader, device="cuda", **eval_dic):
    eval_value_list = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            val_data = batch.to(device)
            result_dic = evaluator.evaluate(model, val_data, **eval_dic)
            if torch.is_tensor(result_dic["evaluation"]):
                eval_value_list.append(result_dic["evaluation"].item())
            else:
                eval_value_list.append(result_dic["evaluation"])
        # end for
    avg_eval_value = sum(eval_value_list) / len(eval_value_list)
    return avg_eval_value


def load_numbers_from_file_and_compute_mean(file: str, *args, **kwargs):
    """
    file: path to file
        each line in file contains only one number.
    """
    s = 0.0
    count = 0
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            s += float(line)
            count += 1
        print(s / count)


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed
