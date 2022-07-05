# ------python import------------------------------------
import copy
import os
import warnings

import numpy as np
import torch
import wandb

# ----------- parse arguments----------------------------------
from CheXpert2.Parser import init_parser
from CheXpert2.custom_utils import Experiment, set_parameter_requires_grad
from CheXpert2.dataloaders.chexpertloader import chexpertloader
# -----local imports---------------------------------------
from CheXpert2.models.CNN import CNN
from CheXpert2.models.Unet import Unet
from CheXpert2.training.training import training

# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def main():

    # -------- proxy config ---------------------------
    from six.moves import urllib

    proxy = urllib.request.ProxyHandler(
        {
            "https": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
            "http": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
        }
    )
    os.environ["HTTPS_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
    os.environ["HTTP_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
    # construct a new opener using your proxy settings
    opener = urllib.request.build_opener(proxy)
    # install the openen on the module-level
    urllib.request.install_opener(opener)

    # ------------ parsing & Debug -------------------------------------
    parser = init_parser()
    args = parser.parse_args()

    os.environ["DEBUG"] = "False"
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = "data/CheXpert-v1.0-small"
    # ----------- hyperparameters-------------------------------------<
    config = {
        # AdamW
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 0.001,
        "weight_decay": 0.01,
        # loss and optimizer
        "optimizer": torch.optim.AdamW,
        "criterion": torch.nn.BCEWithLogitsLoss(),
        # RandAugment
        "N": 2,
        "M": 9,
        "clip_norm": 5
    }
    config = config | vars(args)
    wandb.init(
        project="Chestxray", entity="ccsmtl2", config=copy.copy(config)
    )
    
    config = wandb.config
    # ---------- Sampler -------------------------------------------
    from Sampler import Sampler

    Sampler = Sampler(f"{img_dir}/train.csv")
    if not config["sampler"]:
        Sampler.samples_weight = torch.ones_like(
            Sampler.samples_weight
        )  # set all weights equal
    # ------- device selection ----------------------------
    if torch.cuda.is_available():
        device = f"cuda:{config['device']}" if config['device'] != "parallel" else "cuda:0"

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    print("The model has now been successfully loaded into memory")

    # -----------model initialisation------------------------------
    
    model = CNN(config["model"], 13, freeze_backbone=False)
    model = torch.nn.DataParallel(model)
    # send model to gpu
    model = model.to(device, memory_format=torch.channels_last)

    # -------data initialisation-------------------------------

    train_dataset = chexpertloader(
        f"{img_dir}/train.csv",
        img_dir=img_dir,
        img_size=config["img_size"],
        prob=config["augment_prob"],
        intensity=config["augment_intensity"],
        label_smoothing=config["label_smoothing"],
        cache=False,
        num_worker=os.cpu_count(),
        unet=False,
        channels=3,
        N=config["N"],
        M=config["M"],
    )
    val_dataset = chexpertloader(
        f"{img_dir}/valid.csv",
        img_dir=img_dir,
        img_size=config["img_size"],
        cache=False,
        num_worker=os.cpu_count(),
        unet=False,
        channels=3,
    )

    # rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    # batch_size : maximum possible without crashing

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=Sampler.sampler(),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    print("The data has now been loaded successfully into memory")

    # ------------- Metrics & Trackers -----------------------------------------------------------
   
    wandb.watch(model)
    
    experiment = Experiment(
        f"{config['model']}", is_wandb=True, tags=None, config=copy.copy(config)
    )

    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug
    import yaml
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]
    metric = Metrics(num_classes=13,names=names, threshold=np.zeros((13)) + 0.5)
    metrics = metric.metrics()

    # ------------training--------------------------------------------
    print("Starting training now")

    # initialize metrics loggers
    optimizer = config["optimizer"](model.parameters())
    print(model._get_name())
    results,summary = training(
        model,
        optimizer,
        config["criterion"],
        training_loader,
        validation_loader,
        device,
        minibatch_accumulate=1,
        epoch_max=50,
        patience=10,
        experiment=experiment,
        metrics=metrics,
        clip_norm= config["clip_norm"]
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
    # 2) roc curves
    #     experiment.log_metric(
    #         "roc_curves",
    #         wandb.plot.roc_curve(
    #             convert(results[0]),
    #             convert(results[1]),
    #             labels=names[:-1],
    #
    #         ),
    #         epoch=None
    #     )
    #TODO : define our own roc curves to plot on wandb
        for key,value in summary.items() :
            wandb.run.summary[key] = value


if __name__ == "__main__":
    main()
