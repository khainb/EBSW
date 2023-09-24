import torch


class Saver:
    def __init__(self):
        super().__init__()

    @staticmethod
    def save_checkpoint(ae, optimizer, path, **kwargs):
        dic = {"autoencoder": ae.state_dict(), "optimizer": optimizer.state_dict()}
        if "scheduler" in kwargs.keys():
            dic["scheduler"] = kwargs["scheduler"].state_dict()
        if "f_scheduler" in kwargs.keys():
            dic["f_scheduler"] = kwargs["f_scheduler"].state_dict()
        torch.save(dic, path)
        return

    @staticmethod
    def save_best_weights(ae, path):
        torch.save(ae.state_dict(), path)
        return


class PreprocessedDataSaver:
    def __init__(self):
        super().__init__()

    def save(self, path, data_dict):
        return


class GeneralSaver:
    def __init__(self):
        super().__init__()

    @staticmethod
    def save_checkpoint(trained_model, optimizer, path, model_name, **kwargs):
        """
        model_name: str
            "classifier"|"autoencoder"
        """
        dic = {model_name: trained_model.state_dict(), "optimizer": optimizer.state_dict()}
        if "scheduler" in kwargs.keys():
            dic["scheduler"] = kwargs["scheduler"].state_dict()
        torch.save(dic, path)
        return

    @staticmethod
    def save_best_weights(trained_model, path):
        torch.save(trained_model.state_dict(), path)
        return
