import argparse
import json
import os
import os.path as osp
import random
import shutil
import statistics
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from add_noise_to_data.random_noise import RandomNoiseAdder
from dataset.modelnet40 import ModelNet40
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from loss import EMD, SWD, Chamfer
from models import PointCapsNet, PointNetAE
from tqdm import tqdm
from utils import load_model_for_evaluation


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--logdir", type=str, help="folder contains weights")
    parser.add_argument("--data_path", default="dataset/modelnet40_ply_hdf5_2048/", type=str, help="path to data")
    args = parser.parse_args()
    config = args.config
    logdir = args.logdir
    data_path = args.data_path
    args = json.load(open(config))

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    # save_results folder
    save_folder = osp.join(logdir, args["save_folder"])
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(">Save_folder was created successfully at:", save_folder)
    else:
        print(">Folder {} is existing.".format(save_folder))
        print(">Do you want to remove it?")
        answer = None
        while answer not in ("yes", "no"):
            answer = input("Enter 'yes' or 'no': ")
            if answer == "yes":
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
            elif answer == "no":
                print("SOME FILES WILL BE OVERWRITTEN OR APPENDED.")
                print("If you do not want this, please stop during next 30s.")
                time.sleep(30)
            else:
                print("Please enter yes or no.")
    fname = osp.join(save_folder, "config.json")
    with open(fname, "w") as fp:
        json.dump(args, fp, indent=4)

    # print hyperparameters
    print("You have 5s to check the hyperparameters below.")
    print(args)
    time.sleep(5)

    # device
    device = torch.device(args["device"])

    # NoiseAdder
    if args["add_noise"]:

        if args["noise_adder"] == "random":
            noise_adder = RandomNoiseAdder(mean=args["mean_noiseadder"], std=args["std_noiseadder"])
        else:
            raise ValueError("Unknown noise_adder type.")

    # dataloader
    if args["dataset_type"] == "shapenet55":
        print(data_path)
        dset = ShapeNetCore55XyzOnlyDataset(data_path, args["num_points"], "test")  # root is a npz file
    elif args["dataset_type"] == "modelnet40":
        dset = ModelNet40(data_path, num_points=args["num_points"])  # root is a folder containing h5 file
    else:
        raise ValueError("Unknown dataset type.")

    loader = data.DataLoader(
        dset,
        batch_size=1,
        pin_memory=True,
        num_workers=args["num_workers"],
        shuffle=args["shuffle"],
        worker_init_fn=seed_worker,
    )

    # distance
    chamfer = Chamfer()
    swd = SWD(args["num_projs"], device)
    emd = EMD()

    # architecture
    if args["architecture"] == "pointnet":
        ae = PointNetAE(
            args["embedding_size"],
            args["input_channels"],
            args["input_channels"],
            args["num_points"],
            args["normalize"],
        ).to(device)

    elif args["architecture"] == "pcn":
        ae = PointCapsNet(
            args["prim_caps_size"],
            args["prim_vec_size"],
            args["latent_caps_size"],
            args["latent_vec_size"],
            args["num_points"],
        ).to(device)

    else:
        raise ValueError("Unknown architecture.")

    try:
        ae = load_model_for_evaluation(ae, args["model_path"])
    except:
        try:
            ae = load_model_for_evaluation(ae, osp.join(logdir, args["model_path"]))
        except:
            in_dic = {"key": "autoencoder"}
            ae = load_model_for_evaluation(ae, osp.join(logdir, args["model_path"]), **in_dic)

    # test
    chamfer_list = []
    swd_list = []
    emd_list = []

    ae.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader)):
            if args["dataset_type"] in ["modelnet40"]:
                batch = batch[0].to(device)
            else:
                batch = batch.to(device)

            inp = batch.detach().clone()

            if args["add_noise"]:
                args["test_origin"] = True
                batch = noise_adder.add_noise(batch)

            try:
                reconstruction = ae.decode(ae.encode(batch))
            except:
                latent, reconstruction = ae.forward(batch)

            if args["test_origin"]:
                reconstruction, batch = reconstruction[:, :, :3], inp

            chamfer_list.append(chamfer.forward(reconstruction.contiguous(), batch)["loss"].item())
            swd_list.append(np.sqrt(swd.forward(reconstruction, batch)["loss"].item()))

            try:
                emd_list.append(emd.forward(reconstruction, batch)["loss"].item())
            except:
                # To fix bug: RuntimeError: set_d.is_contiguous() INTERNAL ASSERT FAILED at src/structural_loss.cpp:30
                reconstruction = reconstruction.contiguous()
                batch = batch.contiguous()
                emd_list.append(emd.forward(reconstruction, batch)["loss"].item())
    # report
    test_log = osp.join(save_folder, "reconstruction_test.log")
    _log = "{}| stddev {}| mean {}\n"

    plt.hist(chamfer_list, density=False, bins=args["bin"])
    plt.ylabel("Frequency")
    plt.xlabel("Chamfer value")
    save_path = osp.join(save_folder, "chamfer.eps")
    plt.savefig(save_path)
    plt.clf()

    plt.hist(swd_list, density=False, bins=args["bin"])
    plt.ylabel("Frequency")
    plt.xlabel("SW value")
    save_path = osp.join(save_folder, "swd.eps")
    plt.savefig(save_path)
    plt.clf()

    plt.hist(emd_list, density=False, bins=args["bin"])
    plt.ylabel("Frequency")
    plt.xlabel("EMD value")
    save_path = osp.join(save_folder, "emd.eps")
    plt.savefig(save_path)
    plt.clf()

    log = _log.format("chamfer", statistics.stdev(chamfer_list), statistics.mean(chamfer_list))
    with open(test_log, "a") as fp:
        fp.write(log)
        print(log)

    log = _log.format("swd", statistics.stdev(swd_list), statistics.mean(swd_list))
    with open(test_log, "a") as fp:
        fp.write(log)
        print(log)

    log = _log.format("emd", statistics.stdev(emd_list), statistics.mean(emd_list))
    with open(test_log, "a") as fp:
        fp.write(log)
        print(log)


if __name__ == "__main__":
    main()
