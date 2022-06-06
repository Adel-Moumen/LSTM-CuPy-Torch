import torch
import torch.nn as nn
import time

import config

import os
import pandas as pd


def train(dataset, model, device, name, seq_len):
    """
    :param dataset: BenchmarkTrain Object
    :param model: RNNs model
    :param device: Cuda or CPU
    :param name: name of the model (for csv)
    :param seq_len: current time_step
    """
    model.train()

    for data in dataset:
        # load x & y sets
        x_train = data["X"]
        y_train = data["Y"]

        # to device
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        mean_inference = 0
        mean_backward = 0
        for i in range(config.ITERATOR + 2):

            torch.cuda.synchronize()

            # inference
            time1 = time.time()
            pred, _ = model(x_train)
            torch.cuda.synchronize()
            time2 = time.time()

            loss = nn.MSELoss()(pred, y_train)

            # compute backward
            time3 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            time4 = time.time()

            if i >= 2:
                print(i)
                mean_inference += time2 - time1
                mean_backward += time4 - time3

        with torch.no_grad():
            liste = [name, seq_len, mean_inference / config.ITERATOR, mean_backward / (config.ITERATOR)]
            print("{} = Inference time : {} Training Time : {}".format(
                name,
                mean_inference / (config.ITERATOR),
                mean_backward / (config.ITERATOR))
            )

            # if csv not create we create our and feed the first line with our values
            if not os.path.isfile(config.CSV_PATH):
                df = pd.DataFrame(columns=["Model", "Time Step", "Time Inference", "Time Training"])
                df.loc[1] = liste
                df.to_csv(config.CSV_PATH, index=False)
            # if csv then we append a new row with our values
            else:
                df = pd.read_csv(config.CSV_PATH)
                df.loc[len(df) + 1] = liste
                df.to_csv(config.CSV_PATH, index=False)