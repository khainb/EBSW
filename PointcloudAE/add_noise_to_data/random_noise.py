import os.path as osp
import sys

import torch


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from add_noise_to_data.interface import NoiseAdd_er


class RandomNoiseAdder(NoiseAdd_er):
    def __init__(self, mean=0.0, std=0.1, **kwargs):
        super().__init__()
        self.mean = mean
        self.std = std

    def add_noise(self, data, **kwargs):
        noise = torch.normal(self.mean, self.std, size=data.shape).to(data.device)
        perturbed_data = data + noise
        return perturbed_data


if __name__ == "__main__":
    data = torch.empty((1, 2048, 3)).uniform_(0, 1).cuda()
    noise_adder = RandomNoiseAdder()
    perturbed_data = noise_adder.add_noise(data)
    print(perturbed_data.shape)
