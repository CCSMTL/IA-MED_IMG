# ------python import------------------------------------
import os
import urllib
import warnings

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data.sampler import SequentialSampler

import wandb
from CheXpert2.Experiment import Experiment
# ----------- parse arguments----------------------------------
from CheXpert2.Parser import init_parser
from CheXpert2.custom_utils import set_parameter_requires_grad
from CheXpert2.dataloaders.CXRLoader import CXRLoader
# -----local imports---------------------------------------
from CheXpert2.models.CNN import CNN
from CheXpert2.training.training import training

# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# torch.backends.cudnn.benchmark = True

try:
    os.environ["img_dir"] = os.environ["img_dir"]
except:
    os.environ["img_dir"] = ""
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

    config = vars(args)
    torch.set_num_threads(config["num_worker"])

    # ----------- load classes ----------------------------------------
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]

    #--------- set up augment_prob ---------------------------------
    if len(config["augment_prob"]) == 1:
        prob = [0, ] * 5
        for i in range(5):
            prob[i] = config[f"augment_prob_{i}"]
    else:
        prob = config["augment_prob"]

    # --------- instantiate experiment tracker ------------------------
    experiment = Experiment(
        f"{config['model']}", names=names, tags=None, config=config, epoch_max=config["epoch"], patience=10,
        no_log=False
    )

    if dist.is_initialized():
        dist.barrier()
        torch.cuda.device(device)
    else:
        config = wandb.config
    print(config["augment_prob"])

    return config, img_dir, experiment, device, prob,names


def main(config, img_dir, model, experiment, optimizer, criterion, device, prob, metrics, pretrain=False):
    # -------data initialisation-------------------------------


    if os.environ["DEBUG"] =="False" :
        num_samples = 80000
    else :
        num_samples=100

    train_dataset = CXRLoader(
        split="Train",
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
        pretrain=pretrain
    )
    val_dataset = CXRLoader(
        split="Valid",
        img_dir=img_dir,
        img_size=config["img_size"],
        cache=False,
        num_worker=config["num_worker"],
        unet=False,
        channels=config["channels"],
        N=0,
        M=0,
        pretrain=pretrain
    )

    if train_dataset.weights is not None :
        sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weights, num_samples=min(num_samples,len(train_dataset)))
    else :
        sampler = torch.utils.data.SubsetRandomSampler(list(range(len(train_dataset)))[0:min(num_samples,len(train_dataset))], generator=None)


    if dist.is_initialized() :
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
    # ------------training--------------------------------------------
    print("Starting training now")

    # initialize metrics loggers

    #TODO : remove pretrain boolean?
    training_loader.dataset.pretrain = pretrain
    validation_loader.dataset.pretrain = pretrain

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
        clip_norm=config["clip_norm"],
        autocast=config["autocast"]
    )

    return results




if __name__ == "__main__":
    config, img_dir, experiment, device, prob, names = initialize_config()
    num_classes = len(names)
    optimizer = torch.optim.AdamW
    # -----------model initialisation------------------------------

    model = CNN(config["model"], num_classes, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"],drop_rate=config["drop_rate"],global_pool=config["global_pool"])
    # send model to gpu
    model = model.to(device, dtype=torch.float)

    print("The model has now been successfully loaded into memory")
    # pre-training
    from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug

    metric = Metrics(num_classes=num_classes, names=experiment.names, threshold=np.zeros((num_classes)) + 0.5)
    metrics = metric.metrics()
    if config["pretraining"] !=0 :
        experiment2 = Experiment(
            f"{config['model']}", names=names, tags=None, config=config, epoch_max=config["pretraining"], patience=5,
            no_log=True
        )
        optimizer = optimizer(
            model.parameters(),
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )

        results = main(config, img_dir, model, experiment2, optimizer, torch.nn.BCEWithLogitsLoss(), device, prob,
                       metrics=metrics, pretrain=False)

        #set_parameter_requires_grad(model.backbone)

    #setting up for the training





    # training

    results = main(config, img_dir, model, experiment, torch.optim.SGD(model.parameters(),lr=config["lr"]), torch.nn.BCEWithLogitsLoss(), device, prob, metrics,
                   pretrain=False)
    experiment.end(results)
