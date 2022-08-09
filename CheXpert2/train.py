# ------python import------------------------------------
import os
import urllib
import warnings

import numpy as np
import torch
import torch.distributed as dist
import yaml

import wandb
from CheXpert2.Experiment import Experiment
# ----------- parse arguments----------------------------------
from CheXpert2.Parser import init_parser
from CheXpert2.dataloaders.Chexpertloader import Chexpertloader
# -----local imports---------------------------------------
from CheXpert2.models.CNN import CNN
from CheXpert2.training.training import training


# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
#torch.autograd.set_detect_anomaly(False)
#torch.autograd.profiler.profile(False)
#torch.autograd.profiler.emit_nvtx(False)
#torch.backends.cudnn.benchmark = True




def initialize_config():
    # -------- proxy config ---------------------------

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
    # 1) set up debug env variable
    os.environ["DEBUG"] = str(args.debug)
    if args.debug:
        os.environ["WANDB_MODE"] = "offline"
    # 2) load from env.variable the data repository location
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = "data"

    # ---------- Device Selection ----------------------------------------
    if torch.cuda.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
        else:
            device = args.device

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    # ----------- hyperparameters-------------------------------------<
    # config = {
    #     # loss and optimizer
    #     "optimizer": "AdamW",
    #     "criterion": "BCEWithLogitsLoss",
    #
    # }

    config = vars(args)

    # optimizer = reduce(getattr, [torch.optim] + config["optimizer"].split("."))
    # criterion = reduce(getattr, [torch.nn] + config["criterion"].split("."))()
    optimizer = torch.optim.AdamW
    criterion = torch.nn.BCELoss()
    torch.set_num_threads(config["num_worker"])

    # ----------- load classes ----------------------------------------
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]

    if len(config["augment_prob"]) == 1:
        prob = [0, ] * 5
        for i in range(5):
            prob[i] = config[f"augment_prob_{i}"]
    else:
        prob = config["augment_prob"]

    # --------- instantiate experiment tracker ------------------------
    experiment = Experiment(
        f"{config['model']}", names=names, tags=None, config=config, epoch_max=config["epoch"], patience=5
    )

    if dist.is_initialized():
        dist.barrier()
        torch.cuda.device(device)
    else:
        config = wandb.config
    print(config["augment_prob"])
    from CheXpert2.Sampler import Sampler
    Sampler = Sampler(f"{img_dir}/train.csv")
    sampler = Sampler.sampler()
    return config, img_dir, experiment, optimizer, criterion, device, prob, sampler


def main(config, img_dir, experiment, optimizer, criterion, device, prob, sampler):
    # ---------- Sampler -------------------------------------------

    # -----------model initialisation------------------------------

    model = CNN(config["model"], 13, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"])
    # send model to gpu
    model = model.to(device, dtype=torch.float)

    print("The model has now been successfully loaded into memory")
    # -------data initialisation-------------------------------

    train_dataset = Chexpertloader(
        f"{img_dir}/train.csv",
        img_dir=img_dir,
        img_size=config["img_size"],
        prob=prob,
        intensity=config["augment_intensity"],
        label_smoothing=config["label_smoothing"],
        cache=config["cache"],
        num_worker=config["num_worker"],
        unet=False,
        channels=config["channels"],
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
        channels=config["channels"],
    )



    optimizer = optimizer(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    if dist.is_initialized():  # use of multiple gpu
        from torch.utils.data.sampler import SequentialSampler
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        local_rank = int(os.environ['LOCAL_RANK'])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler))

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        pin_memory=True,
        sampler=sampler,

    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        pin_memory=True,
        shuffle=False,
    )
    print("The data has now been loaded successfully into memory")

    # ------------- Metrics & Trackers -----------------------------------------------------------

    experiment.watch(model)

    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug
    metric = Metrics(num_classes=13, names=experiment.names, threshold=np.zeros((13)) + 0.5)
    metrics = metric.metrics()

    # ------------training--------------------------------------------
    print("Starting training now")

    # initialize metrics loggers

    results = training(
        model,
        optimizer,
        criterion,
        training_loader,
        validation_loader,
        device,
        minibatch_accumulate=1,
        experiment=experiment,
        metrics=metrics,
        clip_norm=config["clip_norm"]
    )

    experiment.end(results)


if __name__ == "__main__":
    config, img_dir, experiment, optimizer, criterion, device, prob, sampler = initialize_config()
    main(config, img_dir, experiment, optimizer, criterion, device, prob, sampler)
