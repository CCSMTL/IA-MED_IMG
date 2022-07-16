import contextlib
import os
import pathlib

import torch

import wandb


# -----------------------------------------------------------------------------------
class Experiment:
    def __init__(self, directory, is_wandb=False, tags=None, config=None):
        self.is_wandb = is_wandb
        self.directory = "log/" + directory
        self.weight_dir = "models/models_weights/" + directory
        self.rank = 0
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()

        if tags is not None:
            self.directory += f"/{tags[0]}"
            self.weight_dir += f"/{tags[0]}"

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        path = pathlib.Path(self.weight_dir)
        path.mkdir(parents=True, exist_ok=True)

        root, dir, files = list(os.walk(self.directory))[0]

        for f in files:
            os.remove(root + "/" + f)

        # if config is not None:
        #     for key, value in config.items():
        #         config[key] = str(value)
        #     with open(f"{self.directory}/config.json", "w") as f:
        #         json.dump(config, f)
        if is_wandb:
            wandb.init(project="Chestxray", entity="ccsmtl2", config=config)

    def log_metric(self, metric_name, value, epoch):
        if self.rank == 0:

            f = open(f"{self.directory}/{metric_name}.txt", "a")
            if type(value) == list:
                f.write("\n".join(str(item) for item in value))
            else:
                f.write(f"{epoch} , {str(value)}")

            if self.is_wandb:
                wandb.log({metric_name: value})
            else:
                print({metric_name: value})

    def save_weights(self, model):
        if self.rank == 0:
            if not os.environ["DEBUG"] == "True":
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
