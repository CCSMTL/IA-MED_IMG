#------python import------------------------------------
import warnings

import pandas as pd
import torch
import wandb
import os
import argparse
import torchvision
import numpy as np
#-----local imports---------------------------------------
from models.CNN import CNN
from training.training import training
from training.dataloaders.cxray_dataloader import CustomImageDataset
from custom_utils import Experiment,set_parameter_requires_grad




# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# torch.backends.cudnn.benchmark = True

#----------- parse arguments----------------------------------
def init_parser() :
    parser = argparse.ArgumentParser(description='Launch training for a specific model')

    parser.add_argument('--model',
                        default='alexnet',
                        const='all',
                        type=str,
                        nargs='?',
                        choices=torch.hub.list('pytorch/vision:v0.10.0'),
                        required=True,
                        help='Choice of the model')

    parser.add_argument('--img_size',
                        default=320,
                        const='all',
                        type=int,
                        nargs='?',
                        required=False,
                        help='width and length to resize the images to. Choose a value between 320 and 608.')

    parser.add_argument('--wandb',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='do you wish (and did you setup) wandb? You will need to add the project name in the initialization of wandb in train.py')

    parser.add_argument('--epoch',
                        default=50,
                        const='all',
                        type=int,
                        nargs='?',
                        required=False,
                        help="Number of epochs to train ; a patiance of 5 is implemented by default")
    parser.add_argument('--batch_size',
                        default=50,
                        const='all',
                        type=int,
                        nargs='?',
                        required=False,
                        help="The batch size to use. If > max_batch_size,gradient accumulation will be used")

    parser.add_argument('--tags',
                        default=[],
                        const='all',
                        type=list,
                        nargs='?',
                        required=False,
                        help="extra tags to add to the logs")
    parser.add_argument('--debug',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='do you wish  execute small train set in debug mode')

    return parser

def main() :
    parser=init_parser()
    args = parser.parse_args()
    os.environ["DEBUG"] = str(args.debug)
    from Sampler import Sampler
    Sampler=Sampler()
    criterion = torch.nn.BCEWithLogitsLoss()

    # -----------model initialisation------------------------------
    model=CNN(args.model,14)
    n = len([param for param in model.named_parameters()])
    set_parameter_requires_grad(model,n-2)
    max_batch_size=8 # defines the maximum batch_size supported by your gpu for a specific model.
    accumulate=args.batch_size//max_batch_size
    print(f"mini batch size : {max_batch_size}. The gradient will be accumulated {accumulate} times")
    if torch.cuda.is_available():
        device = "cuda:0" # The id of the gpu (e.g. 0 , can change on multi gpu devices)
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    print("The model has now been successfully loaded into memory")


    # -------data initialisation-------------------------------
    #os.environ["WANDB_MODE"] = "offline"

    from Metrics import Metrics

    train_dataset = CustomImageDataset(f"data/training",num_classes=14,img_size=args.img_size,prob=0.1,intensity=0.1,label_smoothing=0.1)
    val_dataset = CustomImageDataset(f"data/validation",num_classes=14,img_size=args.img_size)

    #rule of thumb : num_worker = 4 * number of gpu ; on windows leave =0
    #batch_size : maximum possible without crashing



    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=max_batch_size, num_workers=0,pin_memory=True,sampler=Sampler.sampler())
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(max_batch_size*2), num_workers=0,pin_memory=True,sampler=Sampler.sampler())
    print("The data has now been loaded successfully into memory")
    # ------------training--------------------------------------------
    print("Starting training now")



    #send model to gpu
    model = model.to(device)

    #initialize metrics loggers

    if args.wandb :
        wandb.init(project="test-project", entity="ai-chexnet")
        wandb.watch(model)

    experiment = Experiment(f"{args.model}",is_wandb=args.wandb,tags=args.tags)

    optimizer = torch.optim.AdamW(model.parameters())
    metric=Metrics(num_classes=14,threshold=np.zeros((14))+0.5)
    metrics=metric.metrics()
    training(model,optimizer,criterion,training_loader,validation_loader,device,minibatch_accumulate=accumulate,epoch_max=args.epoch,patience=5,experiment=experiment,metrics=metrics)

if __name__ == "__main__":
     main()