import glob
import os

import numpy as np
import torch.utils.data as data


class Fine3dMatchDataset(data.Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        flistname = os.path.join(root, "*.npz")
        self.flistname = sorted(glob.glob(flistname), key=lambda fname: int(fname.split("/")[-1].split(".npz")[0]))
        self.flist = [np.load(fname, allow_pickle=True)["arr_0"].item() for fname in self.flistname]

    def __len__(self):
        return len(self.flistname)

    def __getitem__(self, i):
        return self.flist[i]["points"], self.flist[i]["features"]
