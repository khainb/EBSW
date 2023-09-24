import glob
import os

import open3d
import torch.utils.data as data


try:
    from dataset.interface import Dataset
except:
    from interface import Dataset


class ThreeDMatchRawDataset(data.Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.flistname = os.path.join(root, "*.ply")
        self.flistname = sorted(glob.glob(self.flistname))
        self.flist = [open3d.io.read_point_cloud(fname) for fname in self.flistname]
        self.fnames = [fname.split("/")[-1].split(".ply")[0].split("cloud_bin_")[-1] for fname in self.flistname]

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, i):
        return self.flist[i], self.fnames[i]


Dataset.register(ThreeDMatchRawDataset)
assert issubclass(ThreeDMatchRawDataset, Dataset)
