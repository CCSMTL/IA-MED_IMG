import contextlib
import copy
import os
import pathlib

import numpy as np
import pandas as pd
import torch

import wandb


# -----------------------------------------------------------------------------------

def convert(array1):
    array = copy.copy(array1)
    answers = []
    array = array.numpy().round(0)
    for item in array:
        if np.max(item) == 0:
            answers.append(13)
        else:
            answers.append(np.argmax(item))
    return answers


class Experiment:
    def __init__(self, directory, names, tags=None, config=None):
        self.names = names
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

        wandb.init(project="Chestxray", entity="ccsmtl2", config=config)

        self.summary = {}
        self.metrics = {}

    def log_metric(self, metric_name, value, epoch):
        if self.rank == 0:
            wandb.log({metric_name: value})
            self.metrics[metric_name] = value

    def save_weights(self, model):
        if self.rank == 0:
            wandb.save(f"{self.weight_dir}/{model._get_name()}.pt")

    def summarize(self):
        self.summary = self.metrics

    def watch(self, model):
        wandb.watch(model)

    def end(self, results):
        if self.rank == 0:
            # 1) confusion matrix

            self.log_metric(
                "conf_mat",
                wandb.sklearn.plot_confusion_matrix(
                    convert(results[0]),
                    convert(results[1]),
                    self.names,
                ),
                epoch=None
            )

            df = pd.read_csv("../data/chexnet_results.csv", index_col=0)
            df.columns = ["chexnet", "chexpert"]

            df["ours"] = self.summary["auc"].values()
            df.fillna(0, inplace=True)

            import plotly.graph_objects as go
            fig = go.Figure()
            for column in ["chexnet", "chexpert", "ours"]:
                # fig.add_line_polar(df,r=column, theta=np.arange(0, 2 * np.pi, 2 * np.pi / 14), color=column,
                #                     line_close=True,
                #                     color_discrete_sequence=px.colors.sequential.Plasma_r,
                #                     template="plotly_dark", )

                fig.add_trace(go.Scatterpolar(
                    r=df[column],
                    theta=self.names,
                    mode='lines',
                    name=column,

                    #    line_color='peru'
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        tick0=0,
                        dtick=0.2, )),
                showlegend=True,
                title='Polar chart',

                # color_discrete_sequence=px.colors.sequential.Plasma_r,
                template="plotly_dark",

            )
            # fig.update_polar(ticktext=names)
            fig.write_image("polar.png")
            wandb.log({"polar_chart": fig})
            for key, value in self.summary.items():
                wandb.run.summary[key] = value

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
