import torch
import visdom
import numpy as np
import argparse
import os

from engine.trainer import train
from config.set_params import params as sp
from modeling.model import HARmodel
from utils.build_dataset import build_dataloader

def main():
    """Driver file for training HAR model."""

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    params = sp().params

    model = HARmodel(params["input_dim"], params["num_classes"])

    if params["use_cuda"]:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["lr"],
                                momentum=params["momentum"],
                                weight_decay=params["weight_decay"])
    
    params["start_epoch"] = 1

    # If checkpoint path is given, load checkpoint data
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))

            checkpoint = torch.load(args.checkpoint)
            params["start_epoch"] = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.checkpoint, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(params["resume"]))

    train_loader, val_loader = build_dataloader(params["root"], params)

    logger = visdom.Visdom()

    train(train_loader=train_loader,
          val_loader=val_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          params=params,
          logger=logger,
          )

if __name__ == "__main__":
    main()