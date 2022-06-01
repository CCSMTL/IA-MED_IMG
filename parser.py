import argparse
import torch
def init_parser():
    parser = argparse.ArgumentParser(description="Launch training for a specific model")

    parser.add_argument(
        "--model",
        default="alexnet",
        const="all",
        type=str,
        nargs="?",
        choices=torch.hub.list("pytorch/vision:v0.10.0")
        + torch.hub.list("facebookresearch/deit:main"),
        required=True,
        help="Choice of the model",
    )

    parser.add_argument(
        "--img_size",
        default=320,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="width and length to resize the images to. Choose a value between 320 and 608.",
    )

    parser.add_argument(
        "--device",
        default="0",
        const="all",
        choices=["parallel","0","1"],
        type=str,
        nargs="?",
        required=False,
        help="GPU on which to execute your code. Parallel to use all available gpus",
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do you wish (and did you setup) wandb? You will need to add the project name in the initialization of wandb in train.py",
    )

    parser.add_argument(
        "--epoch",
        default=50,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="Number of epochs to train ; a patiance of 5 is implemented by default",
    )
    parser.add_argument(
        "--batch_size",
        default=50,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="The batch size to use. If > max_batch_size,gradient accumulation will be used",
    )
    parser.add_argument(
        "--accumulate",
        default=1,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="The number of epoch to accumulate gradient. Choose anything >0",
    )
    parser.add_argument(
        "--num_worker",
        default=8,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="The number of process to use to retrieve the data. Please do not exceed 16",
    )

    parser.add_argument(
        "--tags",
        default=None,
        nargs="+",
        required=False,
        help="extra tags to add to the logs",
    )

    parser.add_argument(
        "--frozen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do you wish  to freeze the backbone?",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do you wish to execute small train set in debug mode",
    )

    parser.add_argument(
        "--sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do you wish to run with a sampler (1/n)",
    )

    return parser