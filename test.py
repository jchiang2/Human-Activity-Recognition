import torch
import visdom
import argparse
import os
import pandas as pd

from config.set_params import params as sp
from modeling.model import HARmodel
from utils.preprocessing import HARdataset

def build_testset(params):
    df = pd.read_csv(params["test"], low_memory=False)
    parts = ["belt", "arm", "dumbbell", "forearm"]
    variables = ["roll_{}", "pitch_{}", "yaw_{}", "total_accel_{}", 
                 "accel_{}_x", "accel_{}_y", "accel_{}_z", "gyros_{}_x",
                                             "gyros_{}_y", "gyros_{}_z"]
    var_list = []
    for part in parts:
        for var in variables:
            var_list.append(list(df[var.format(part)]))
    var_list = torch.tensor(var_list)

    return var_list

def main():
    """Driver file to run inference on test data."""
    className = ["A","B","C","D","E"]

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    assert type(args.checkpoint) is str, "Please input path to checkpoint"

    params = sp().params

    model = HARmodel(params["input_dim"], params["num_classes"])

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(args.checkpoint))

    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        return

    dataset = HARdataset(params["root"])
    mean, std = dataset.mean, dataset.std

    logger = visdom.Visdom()

    testset = build_testset(params)
    testset = (testset - mean) / std

    results = []
    for i in range(testset.size(1)):
        test_data = testset[:,i].view(1, 1, -1)
        output = model(test_data)
        results.append(int(output.max(1)[1]))

    results = [className[i] for i in results]
    print("Prediction results:")
    print(results)

if __name__ == "__main__":
    main()

