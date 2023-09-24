import numpy as np
import torch.utils.data as data


class ShapeNetCore55XyzOnlyDataset(data.Dataset):
    def __init__(self, root, num_points=2048, phase="train"):
        super(ShapeNetCore55XyzOnlyDataset, self).__init__()
        assert root.endswith(".npz"), "root must be .npz file"
        assert phase in ["train", "test"]
        self.data = np.load(root)["data"]
        self.num_points = num_points
        self.phase = phase

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        pc = self._pc_normalize(self.data[i])
        if self.phase == "train":
            choice = np.random.choice(pc.shape[0], self.num_points, replace=True)
            pc = pc[choice, :]
        return pc

    def _pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


class ShapeNetCore55_n_attributes(data.Dataset):
    def __init__(self, root, num_attr=12, num_points=2048, phase="train"):
        super(ShapeNetCore55_n_attributes, self).__init__()
        assert root.endswith(".npz"), "root must be .npz file"
        assert phase in ["train", "test"]
        self.data = np.load(root)["data"]
        self.num_points = num_points
        self.phase = phase
        assert num_attr % 3 == 0, "num_attr must be divisible by 3"
        self.num_attr = num_attr

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        pc = self._pc_normalize(self.data[i])
        if self.phase == "train":
            choice = np.random.choice(pc.shape[0], self.num_points, replace=True)
            pc = pc[choice, :]
        _pc = pc
        for i in range(2, int(self.num_attr % 3 + 1)):
            _pc = np.concatenate((_pc, np.power(pc, i)))
        return _pc

    def _pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
