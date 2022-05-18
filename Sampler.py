# TODO : verify SAMPLE WEIGHT

import pandas as pd
import numpy as np
import os
import torch


class Sampler:
    def __init__(self):
        data = pd.read_csv("data/data_table.csv")
        data = data[data["assignation"] == "training"]
        if os.environ["DEBUG"] == "True":
            data = data.iloc[0:100]
        names = [
            "Cardiomegaly",
            "Emphysema",
            "Effusion",
            "Consolidation",
            "Hernia",
            "Infiltration",
            "Mass",
            "Nodule",
            "Atelectasis",
            "Pneumothorax",
            "Pleural_Thickening",
            "Pneumonia",
            "Fibrosis",
            "Edema",
            "No Finding",
        ]
        count = []

        for name in names:
            count.append(np.sum(data[name]))
        count = 1 / np.array(count)
        count[np.isinf(count)] = 0

        count = torch.nn.functional.softmax(torch.tensor(count))
        m = data[names].values.T
        m = np.vstack([m, np.ones_like(m[0])])
        classes = np.argmax(m, axis=0)

        for i in range(0, 15):
            classes = np.where(classes == i, count[i], classes)

        self.samples_weight = torch.tensor(classes)

    def sampler(self):

        return torch.utils.data.sampler.WeightedRandomSampler(
            self.samples_weight, len(self.samples_weight)
        )
