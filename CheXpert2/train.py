# ------python import------------------------------------
import copy
import os
import warnings
from functools import reduce

import numpy as np
import torch
import torch.distributed as dist

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


def cleanup():
    torch.distributed.destroy_process_group()


#
# def init_distributed(rank, world_size):
#     """
#     “node” is a system in your distributed architecture. In lay man’s terms, a single system that has multiple GPUs can be called as a node.
#
#     “global rank” is a unique identification number for each node in our architecture.
#
#     “local rank” is a unique identification number for processes in each node.
#
#     “world” is a union of all of the above which can have multiple nodes where each node spawns multiple processes. (Ideally, one for each GPU)
#
#     “world_size” is equal to number of nodes * number of gpus
#     """
#     # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
#     dist_url = "env://"  # default
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '5000'
#
#     # only works with torch.distributed.launch // torch.run
#     rank = rank  # id per process?
#     world_size = world_size  # number of processes
#     local_rank = 0  # int(os.environ['LOCAL_RANK']) #keep to 1 : gpu per process
#
#     # this will make all .cuda() calls work properly
#     torch.cuda.set_device(local_rank)
#     torch.manual_seed(42)
#
#
#
#     torch.distributed.init_process_group(
#             backend="gloo",  # nccl , gloo, etc
#             init_method=dist_url,
#             world_size=world_size,
#             rank=rank)
#      # synchronizes all the threads to reach this point before moving on
#     n = torch.cuda.device_count() // world_size
#     device_ids = list(range(local_rank * n, (local_rank + 1) * n))
#
#     print(
#         f"[{os.getpid()}] rank = {dist.get_rank()}, "
#         + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
#     )
#     torch.distributed.barrier()

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
    os.environ["DEBUG"] = str(args.debug)
    if args.debug:
        os.environ["WANDB_MODE"] = "offline"

    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = "data"
    # ----------- hyperparameters-------------------------------------<
    config = {
        # AdamW
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 0.001,
        "weight_decay": 0.01,
        # loss and optimizer
        "optimizer": "AdamW",
        "criterion": "BCEWithLogitsLoss",
        # RandAugment
        "N": 2,
        "M": 9,
        "clip_norm": 5
    }

    config = config | vars(args)
    wandb.init(project="Chestxray", entity="ccsmtl2", config=config)

    config = wandb.config
    experiment = Experiment(
        f"{config['model']}", is_wandb=True, tags=None, config=config
    )
    optimizer = reduce(getattr, [torch.optim] + config["optimizer"].split("."))
    criterion = reduce(getattr, [torch.nn] + config["criterion"].split("."))()
    if torch.cuda.is_available():
        if -1 in config["device"]:
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
        else:
            device = f"cuda:"
            for i in config['device']:
                device = device + "," + str(i)

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    torch.cuda.device(device)
    torch.set_num_threads(config["num_worker"])
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

    # ----------- parallelisation -------------------------------------

    optimizer = optimizer(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    # -------data initialisation-------------------------------

    train_dataset = Chexpertloader(
        f"{img_dir}/train.csv",
        img_dir=img_dir,
        img_size=config["img_size"],
        prob=config["augment_prob"],
        intensity=config["augment_intensity"],
        label_smoothing=config["label_smoothing"],
        cache=False,
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

    if config["device"] == "parallel":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        local_rank = int(os.environ['LOCAL_RANK'])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, num_replicas=world_size,
    #                                                               rank=rank, shuffle=True)
    # else:
    sampler = Sampler.sampler()
    # rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    # batch_size : maximum possible without crashing

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

    wandb.watch(model)

    import yaml
    with open("data/data.yaml", "r") as stream:
        names = yaml.safe_load(stream)["names"]
    if os.environ["DEBUG"] == "False":
        from CheXpert2.Metrics import Metrics  # sklearn f**ks my debug
        metric = Metrics(num_classes=13, names=names, threshold=np.zeros((13)) + 0.5)
        metrics = metric.metrics()
    else:
        metrics = None

    # ------------training--------------------------------------------
    print("Starting training now")

    # initialize metrics loggers
    print(model._get_name())
    results, summary = training(
        model,
        optimizer,
        criterion,
        training_loader,
        validation_loader,
        device,
        minibatch_accumulate=1,
        epoch_max=config["epoch"],
        patience=10,
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
    dist.init_process_group("nccl")
    main()

    cleanup()
