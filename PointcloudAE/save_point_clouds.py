import json
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as data
from dataset.modelnet40 import ModelNet40
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from models import PointNetAE
from tqdm import tqdm
from utils import load_model_for_evaluation


# Parameters
def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(args):
    # Set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CONFIG = "reconstruction/config.json"
RENDER_PATH = "render"
DATA_PATH = "dataset/shapenet_core55/shapenet57448xyzonly.npz"
args = json.load(open(CONFIG))
args["seed"] = 123
args["dataset_type"] = "shapenetcore55"
device = torch.device(args["device"])
NUM_SAMPLES = 50

# Dataset
print("Start creating dataloader.")
if args["dataset_type"] == "modelnet40":
    dataset = ModelNet40(args["root"], num_points=args["num_points"])
elif args["dataset_type"] == "shapenetcore55":
    dataset = ShapeNetCore55XyzOnlyDataset(DATA_PATH, num_points=args["num_points"], phase="test")
else:
    raise ValueError("Dataset is not supported!")

set_seed(args)
loader = data.DataLoader(
    dataset,
    batch_size=1,
    pin_memory=True,
    num_workers=args["num_workers"],
    # shuffle=args["shuffle"],
    shuffle=True,
    worker_init_fn=seed_worker,
)
print("Finish creating dataloader.")

# Extract samples
raw = []

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(loader)):
        if batch_idx >= NUM_SAMPLES:
            break
        if args["dataset_type"] in ["modelnet40"]:
            raw.append(batch[0])
        else:
            raw.append(batch)

raw = torch.cat(raw)
print(raw.size())

# Save raw samples
FILE_NAME = "reconstruct_random_{}_{}.npy".format(NUM_SAMPLES, args["dataset_type"])
save_path = osp.join(RENDER_PATH, "raw")
os.makedirs(save_path, exist_ok=True)
save_file = osp.join(save_path, FILE_NAME)
np.save(save_file, raw)


# Load model and save reconstruction
def save_reconstruction(samples, model_name):
    print(model_name)
    model_path = f"logs/{model_name}/model.pth"

    # architecture
    ae = PointNetAE(
        args["embedding_size"],
        args["input_channels"],
        args["input_channels"],
        args["num_points"],
        args["normalize"],
    ).to(device)

    ae = load_model_for_evaluation(ae, model_path)

    # ae.eval()
    with torch.no_grad():
        reconstruction = ae.decode(ae.encode(samples.to(device)))

    print(reconstruction.size())

    # Save recon
    save_path = osp.join(RENDER_PATH, "images", model_name)
    os.makedirs(save_path, exist_ok=True)
    save_file = osp.join(save_path, FILE_NAME)
    np.save(save_file, reconstruction.cpu())


list_full_models = [
    "swd",
]

for model in list_full_models:
    save_reconstruction(raw, model)
