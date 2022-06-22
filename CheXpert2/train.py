# ------python import------------------------------------
import copy
import os
import warnings

import numpy as np
import torch
import wandb

from CheXpert2.custom_utils import Experiment, set_parameter_requires_grad
from CheXpert2.dataloaders.CxrayDataloader import CxrayDataloader

# -----local imports---------------------------------------
from CheXpert2.models.CNN import CNN
from CheXpert2.models.Unet import Unet
from CheXpert2.training.training import training

# ----------- parse arguments----------------------------------
from parser import init_parser

# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def main():
    img_dir=os.environ["img_dir"]
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
    os.environ["DEBUG"]=str(args.debug)
    # ----------- hyperparameters-------------------------------------

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
    }
    # ---------- Sampler -------------------------------------------
    from Sampler import Sampler

    Sampler = Sampler(img_dir)
    if not args.sampler:
        Sampler.samples_weight = torch.ones_like(
            Sampler.samples_weight
        )  # set all weights equal
    # ------- device selection ----------------------------
    if torch.cuda.is_available():
        device = f"cuda:{args.device}" if args.device != "parallel" else "cuda:0"

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    print("The model has now been successfully loaded into memory")

    # -----------model initialisation------------------------------
    if args.unet:
        model = Unet(args.model, 14)
    else:
        model = CNN(args.model, 14, freeze_backbone=False)

    if args.device == "parallel":
        model = torch.nn.DataParallel(model)

    # remove the gradient for the backbone
    if args.frozen:
        set_parameter_requires_grad(model.backbone)

    # send model to gpu
    model = model.to(device)

    # -------data initialisation-------------------------------

    train_dataset = CxrayDataloader(
        f"{img_dir}/training",
        num_classes=14,
        img_size=args.img_size,
        prob=args.augment_prob,
        intensity=args.augment_intensity,
        label_smoothing=args.label_smoothing,
        cache=args.cache,
        num_worker=args.num_worker,
        unet=args.unet,
        channels=3,
        N=config["N"],
        M=config["M"],
    )
    val_dataset = CxrayDataloader(
        f"{img_dir}/validation",
        num_classes=14,
        img_size=args.img_size,
        cache=args.cache,
        num_worker=args.num_worker,
        unet=args.unet,
        channels=3,
    )

    # rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    # batch_size : maximum possible without crashing

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        pin_memory=True,
        sampler=Sampler.sampler(),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        pin_memory=True,
    )
    print("The data has now been loaded successfully into memory")

    # ------------- Metrics & Trackers -----------------------------------------------------------
    config = config | vars(args)

    if args.wandb:
        wandb.init(
            project="test-project", entity="ai-chexnet", config=copy.copy(config)
        )

        wandb.watch(model)

    experiment = Experiment(
        f"{args.model}", is_wandb=args.wandb, tags=args.tags, config=copy.copy(config)
    )
    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug

    metric = Metrics(num_classes=14, threshold=np.zeros((14)) + 0.5)
    metrics = metric.metrics()

    # ------------training--------------------------------------------
    print("Starting training now")

    # initialize metrics loggers
    optimizer = config["optimizer"](model.parameters())
    print(model._get_name())
    results = training(
        model,
        optimizer,
        config["criterion"],
        training_loader,
        validation_loader,
        device,
        minibatch_accumulate=args.accumulate,
        epoch_max=args.epoch,
        patience=10,
        experiment=experiment,
        metrics=metrics,
    )

    # -------Final Visualization-------------------------------
    # TODO : create Visualization of the best model and upload those to wandb

    def convert(array1):
        array = copy.copy(array1)
        answers = []
        array = array.numpy().round(0)
        for item in array:
            if np.max(item) == 0:
                answers.append(14)
            else:
                answers.append(np.argmax(item))
        return answers

    # from CheXpert2.visualization import

    # 1) confusion matrix
    import yaml

    if wandb.run is not None:
        with open("data/data.yaml", "r") as stream:  # TODO : remove hardcode
            names = yaml.safe_load(stream)["names"]
        experiment.log_metric(
            "conf_mat",
            wandb.sklearn.plot_confusion_matrix(
                convert(results[0]),
                convert(results[1]),
                names,
            ),
            epoch=None
        )
        experiment.log_metric(
            "roc_curves",
            wandb.plot.roc_curve(
                results[0].numpy(),
                results[1].numpy(),
                labels=names,

            ),
            epoch=None
        )

    # 2) roc curves


if __name__ == "__main__":
    main()
