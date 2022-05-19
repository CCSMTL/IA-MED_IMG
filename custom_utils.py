import os
import torch
import pathlib
import wandb
import json

# -----------------------------------------------------------------------------------
class Experiment:
    def __init__(self, directory, is_wandb=False, tags=[], config=None):
        self.is_wandb = is_wandb
        self.directory = "log/" + directory
        self.weight_dir = "models/models_weights/" + directory

        for tag in tags:
            self.directory += f"/{tag}"
            self.weight_dir += f"/{tag}"

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        path = pathlib.Path(self.weight_dir)
        path.mkdir(parents=True, exist_ok=True)

        root, dir, files = list(os.walk(self.directory))[0]

        for f in files:
            os.remove(root + "/" + f)

        if config:
            json.dumps(config, f"{self.directory}/config.json")

    def log_metric(self, metric_name, value, epoch):

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
        if os.environ["DEBUG"] == "False":
            torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")

            if self.is_wandb:
                wandb.save(f"{self.weight_dir}/{model._get_name()}.pt")


# -----------------------------------------------------------------------------------
def set_parameter_requires_grad(model, feature_extract_len):
    for ex, param in enumerate(model.parameters()[::-1]):
        if ex > feature_extract_len:
            param.requires_grad = False
        else:
            param.requires_grad = True


# -----------------------------------------------------------------------------------


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# -----------------------------------------------------------------------------------
