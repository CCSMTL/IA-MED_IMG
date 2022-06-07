import pandas as pd
import numpy as np
import os
import torch
import yaml

with open("data/data.yaml","r") as stream :
    names=yaml.safe_load(stream)["names"]
names+=["No Finding"]
class Sampler:
    def __init__(self):
        data = pd.read_csv("data/data_table.csv")
        data = data[data["assignation"] == "training"]
        if __debug__ :
            data = data.iloc[0:100]

        self.count = []

        for name in names:
            self.count.append(np.sum(data[name]))
        self.count = 1 / np.array(self.count)
        self.count[np.isinf(self.count)] = 0


        m = data[names].values.T
        m = np.vstack([m, np.ones_like(m[0])])
        classes = np.argmax(m, axis=0)

        for i in range(0, 15):
            classes = np.where(classes == i, self.count[i], classes)

        self.samples_weight = torch.tensor(classes)

    def sampler(self):

        return torch.utils.data.sampler.WeightedRandomSampler(
            self.samples_weight, len(self.samples_weight)
        )
