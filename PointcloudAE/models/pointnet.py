import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(PointNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1024, embedding_size)

    def forward(self, x):
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc_mu(x)

        return x


class PointNetDecoder(nn.Module):
    def __init__(self, embedding_size, output_channels=3, num_points=1024):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.output_channels = output_channels
        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_points * output_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        x = x.view(batch_size, self.num_points, self.output_channels)
        x = x.contiguous()
        return x


class PointNetAE(nn.Module):
    def __init__(self, embedding_size=256, input_channels=3, output_channels=3, num_points=1024, normalize=True):
        super(PointNetAE, self).__init__()
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = embedding_size
        self.encoder = PointNetEncoder(embedding_size, input_channels)
        self.decoder = PointNetDecoder(embedding_size, output_channels, num_points)

    def encode(self, x):
        z = self.encoder(x)
        if self.normalize:
            z = F.normalize(z)
        return z

    def decode(self, z):
        y = self.decoder(z)
        return y
