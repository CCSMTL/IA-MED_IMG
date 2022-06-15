import os
import pathlib
import contextlib
import wandb
import json
import torch


# -----------------------------------------------------------------------------------
class Experiment:
    """
    This class is used to log the training information both locally and in wandb.ai
    """

    def __init__(self, directory, is_wandb=False, tags=None, config=None):
        self.is_wandb = is_wandb
        self.directory = "log/" + directory
        self.weight_dir = "models/models_weights/" + directory
        if tags is not None:

            self.directory += f"/{tags[0]}"
            self.weight_dir += f"/{tags[0]}"

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        dirpath = pathlib.Path(self.weight_dir)
        dirpath.mkdir(parents=True, exist_ok=True)

        root, dir, files = list(os.walk(self.directory))[0]

        for file in files:
            os.remove(root + "/" + file)

        if config is not None:
            for key, value in config.items():
                config[key] = str(value)
            with open(f"{self.directory}/config.json", "w") as file:
                json.dump(config, file)

    def log_metric(self, metric_name, value, epoch):

        with open(f"{self.directory}/{metric_name}.txt", "a") as file:
            if isinstance(value, list):
                file.write("\n".join(str(item) for item in value))
            else:
                file.write(f"{epoch} , {str(value)}")

            if self.is_wandb:
                wandb.log({metric_name: value})
            else:
                print({metric_name: value})

    def save_weights(self, model):
        if not __debug__:
            torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")

            if self.is_wandb:
                wandb.save(f"{self.weight_dir}/{model._get_name()}.pt")


# -----------------------------------------------------------------------------------
def set_parameter_requires_grad(model):
    for ex, param in enumerate(model.parameters()):
        param.requires_grad = False


# -----------------------------------------------------------------------------------


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# -----------------------------------------------------------------------------------
@contextlib.contextmanager
def dummy_context_mgr():
    yield None
