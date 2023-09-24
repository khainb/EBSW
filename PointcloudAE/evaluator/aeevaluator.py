import os.path as osp
import sys

import torch


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


class AEEvaluator:
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate(autoencoder, val_data, loss_func, **kwargs):
        autoencoder.eval()
        with torch.no_grad():
            latent = autoencoder.encode(val_data)
            reconstruction = autoencoder.decode(latent)
        _loss = loss_func.forward(val_data, reconstruction, latent=latent, **kwargs)["loss"]
        autoencoder.train()
        return {"evaluation": _loss}
