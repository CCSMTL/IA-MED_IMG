# ------python import------------------------------------
import copy
import os
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

import wandb
# ----------- parse arguments----------------------------------
from CheXpert2.Parser import init_parser
from CheXpert2.custom_utils import Experiment
from CheXpert2.dataloaders.Chexpertloader import Chexpertloader
# -----local imports---------------------------------------
from CheXpert2.models.CNN import CNN
from CheXpert2.training.training import training

# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True




def initialize_config():
    # -------- proxy config ---------------------------

    # proxy = urllib.request.ProxyHandler(
    #     {
    #         "https": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
    #         "http": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
    #     }
    # )
    # os.environ["HTTPS_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
    # os.environ["HTTP_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
    # construct a new opener using your proxy settings
    # opener = urllib.request.build_opener(proxy)
    # install the openen on the module-level
    # urllib.request.install_opener(opener)

    # ------------ parsing & Debug -------------------------------------
    parser = init_parser()
    args = parser.parse_args()
    # 1) set up debug env variable
    os.environ["DEBUG"] = str(args.debug)
    if args.debug:
        os.environ["WANDB_MODE"] = "offline"
    # 2) load from env.variable the data repository location
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = "data"
    # 3) Specify cuda device
    if torch.cuda.is_available():
        if -1 == args.device:
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
        else:
            device = args.device

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    # ----------- hyperparameters-------------------------------------<
    config = {
        # loss and optimizer
        "optimizer": "AdamW",
        "criterion": "BCEWithLogitsLoss",

    }

    config = config | vars(args)

    optimizer = reduce(getattr, [torch.optim] + config["optimizer"].split("."))
    criterion = reduce(getattr, [torch.nn] + config["criterion"].split("."))()


    torch.set_num_threads(config["num_worker"])
    experiment = Experiment(
        f"{config['model']}", is_wandb=device == 0, tags=None, config=config
    )
    if experiment.is_wandb:
        config = wandb.config
        try:
            prob = [0, ] * 5
            for i in range(5):
                prob[i] = config[f"augment_prob_{i}"]
            config["augment_prob"] = prob
        except:
            pass
    if dist.is_initialized():
        dist.barrier()
        torch.cuda.device(device)
    return config, img_dir, experiment, optimizer, criterion, device


def main():
    config, img_dir, experiment, optimizer, criterion, device = initialize_config()
    # ---------- Sampler -------------------------------------------
    from Sampler import Sampler

    Sampler = Sampler(f"{img_dir}/train.csv")
    if not config["sampler"]:
        Sampler.samples_weight = torch.ones_like(
            Sampler.samples_weight
        )  # set all weights equal
    # ------- device selection ----------------------------


    print("The model has now been successfully loaded into memory")

    # -----------model initialisation------------------------------

    model = CNN(config["model"], 13, freeze_backbone=False)
    # send model to gpu
    model = model.to(device, memory_format=torch.channels_last)



    # -------data initialisation-------------------------------

    train_dataset = Chexpertloader(
        f"{img_dir}/train.csv",
        img_dir=img_dir,
        img_size=config["img_size"],
        prob=config["augment_prob"],
        intensity=config["augment_intensity"],
        label_smoothing=config["label_smoothing"],
        cache=config["cache"],
        num_worker=config["num_worker"],
        unet=False,
        channels=3,
        N=config["N"],
        M=config["M"],
    )
    val_dataset = Chexpertloader(
        f"{img_dir}/valid.csv",
        img_dir=img_dir,
        img_size=config["img_size"],
        cache=False,
        num_worker=config["num_worker"],
        unet=False,
        channels=3,
    )

    sampler = Sampler.sampler()

    optimizer = optimizer(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    if dist.is_initialized():
        os.wait(int(dist.get_rank() * 20))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        local_rank = int(os.environ['LOCAL_RANK'])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        from torch.utils.data.sampler import SequentialSampler
        sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler))
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=sampler,

    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=False,
    )
    print("The data has now been loaded successfully into memory")

    # ------------- Metrics & Trackers -----------------------------------------------------------
    if experiment.is_wandb:
        wandb.watch(model)

    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]

    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug
    metric = Metrics(num_classes=13, names=names, threshold=np.zeros((13)) + 0.5)
    metrics = metric.metrics()

    # ------------training--------------------------------------------
    print("Starting training now")

    # initialize metrics loggers

    results, summary = training(
        model,
        optimizer,
        criterion,
        training_loader,
        validation_loader,
        device,
        minibatch_accumulate=1,
        epoch_max=config["epoch"],
        patience=5,
        experiment=experiment,
        metrics=metrics,
        clip_norm=config["clip_norm"]
    )

    # -------Final Visualization-------------------------------
    # TODO : create Visualization of the best model and upload those to wandb

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

    # from CheXpert2.visualization import

    # 1) confusion matrix
    if wandb.run is not None:

        experiment.log_metric(
            "conf_mat",
            wandb.sklearn.plot_confusion_matrix(
                convert(results[0]),
                convert(results[1]),
                names,
            ),
            epoch=None
        )

        df = pd.read_csv("data/chexnet_results.csv", index_col=0)
        df.columns = ["chexnet", "chexpert"]

        df["ours"] = summary["auc"].values()
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
                theta=names,
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
        for key, value in summary.items():
            wandb.run.summary[key] = value


if __name__ == "__main__":
    main()

