import glob
import os.path as osp
import sys

import h5py
import numpy as np
import torch.utils.data as data


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
try:
    from utils import pc_normalize, sample_pc
except:
    from .utils import pc_normalize, sample_pc


class ModelNet40(data.Dataset):
    def __init__(self, root, num_points=2048):
        super().__init__()
        flistname = osp.join(root, "*.h5")
        flistname = sorted(glob.glob(flistname))
        flist = [h5py.File(fname, "r") for fname in flistname]
        self.data = np.concatenate([f["data"][:] for f in flist])
        self.label = np.concatenate([f["label"][:] for f in flist])
        self.num_points = num_points

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, i):
        pc = pc_normalize(self.data[i])
        # pc = sample_pc(pc, self.num_points) #modelnet40 to test autoencoder, so we dont sample new pc from origin pc
        return pc, self.label[i]


class LatentVectorsModelNet40(data.Dataset):
    def __init__(self, root):
        super().__init__()
        assert root.endswith(".npz"), "root must be a npz file"
        self.data = np.load(root)["latent_vectors"]
        self.labels = np.load(root)["labels"]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        return self.data[i].reshape(-1), self.labels[i]


class ModelNet40Train(data.Dataset):
    def __init__(self, root, num_points=2048):
        super().__init__()
        flistname = osp.join(root, "*.h5")
        flistname = sorted(glob.glob(flistname))
        flist = [h5py.File(fname, "r") for fname in flistname]
        self.data = np.concatenate([f["data"][:] for f in flist])
        self.label = np.concatenate([f["label"][:] for f in flist])
        self.num_points = num_points

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, i):
        pc = pc_normalize(self.data[i])
        pc = sample_pc(pc, self.num_points)  # modelnet40 to test autoencoder, so we dont sample new pc from origin pc
        return pc, self.label[i]


class LatentCapsulesModelNet40(data.Dataset):
    def __init__(self, root):
        super().__init__()
        assert root.endswith(".h5"), "root must be a h5 file"
        fp = h5py.File(root)
        self.data = fp["data"]
        self.labels = fp["cls_label"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return self.data[i].reshape(-1), self.labels[i]
