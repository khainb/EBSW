import torch


try:
    from .interface import Trainer
except:
    from interface import Trainer


class AETrainer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def train(ae, loss_func, optimizer, data, **kwargs):
        """
        AUTOENCODER TRAINER
        ae: nn.Module
            the autoencoder with initial weights
        loss_func: nn.Module
            Chamfer, EMD, SWD
        optimizer: nn.optim
            optimizer, such as, SGD or Adam, with initial state
        dataloader: torch.utils.data.dataloader
            dataloader
        device: "cuda" or "cpu"
        """
        ae.train()
        try:
            latent = ae.encode(data)
            reconstructed_data = ae.decode(latent)
        except:
            latent, reconstructed_data = ae(data)

        if "input" in kwargs.keys():
            inp = kwargs["input"]
        else:
            inp = data

        result_dic = loss_func.forward(inp, reconstructed_data.contiguous(), latent=latent, **kwargs)
        optimizer.zero_grad()
        result_dic["loss"].backward()
        optimizer.step()

        result_dic["ae"] = ae
        result_dic["optimizer"] = optimizer
        return result_dic


class LatentCodesGeneratorTrainer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def train(g, ae, loss_func, optimizer, data, **kwargs):
        """
        LATENT CODES GENERATOR TRAINER
        g: nn.Module
            the generator
        ae: nn.Module
            the pretrained autoencoder
        loss_func: a function that returns a dict with at least a key "loss"
            ChamferLoss or ASW
        optimizer: nn.optim
            optimizer for g, such as, SGD or Adam
        data: [batch_size, #points, #point channels]
            a batch of point clouds
        """
        g.train()
        ae.eval()
        with torch.no_grad():
            try:
                latent = ae.encode(data)
            except:
                latent, reconstructed_data = ae.forward(data)

        result_dic = loss_func(g, latent, **kwargs)

        optimizer.zero_grad()
        result_dic["loss"].backward()
        optimizer.step()

        result_dic["g"] = g
        result_dic["optimizer"] = optimizer
        return result_dic


class ClassifierTrainer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def train(classifier, loss_func, optimizer, data, gt_label):
        predicted_label = classifier.forward(data)
        loss = loss_func.forward(predicted_label, gt_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return classifier, optimizer, loss


class adasw_dynamic_eps_AETrainer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def train(ae, loss_func, optimizer, data, epsilon):
        """
        AUTOENCODER TRAINER
        ae: nn.Module
            the autoencoder with initial weights
        loss_func: nn.Module
            ChamferLoss of ASW
        optimizer: nn.optim
            optimizer, such as, SGD or Adam, with initial state
        dataloader: torch.utils.data.dataloader
            dataloader
        device: "cuda" or "cpu"

        """
        ae.train()
        latent_vects = ae.encode(data)
        reconstructed_data = ae.decode(latent_vects)

        loss = loss_func.forward(data, reconstructed_data, epsilon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return ae, optimizer, loss


Trainer.register(AETrainer)
assert issubclass(AETrainer, Trainer)
Trainer.register(ClassifierTrainer)
assert issubclass(ClassifierTrainer, Trainer)
