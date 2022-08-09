import os

import numpy as np
import pandas as pd
import torch


class Sampler:
    def __init__(self, datafolder):
        no_cat = 0

        data = pd.read_csv(datafolder)
        data = data.replace(-1, 0.5).fillna(0)
        self.data = data.iloc[:, 5:19]
        count = self.data.sum().to_numpy()
        count = 1 / count

        weights = np.zeros((len(data)))
        for i, line in data.iterrows():

            vector = line.to_numpy()[5:19]

            weight = count[np.where(np.array(vector) == 1)]
            n = len(weight)
            if n > 1:
                weight = weight[np.random.randint(0, n)]

            elif n == 0:
                weight = count[np.random.randint(0,
                                                 14)]  # assigns random weight to each sample that does not have defined category
                # print("No category had been identified!")
                no_cat += 1

            weights[i] = weight

        print(f"A total of {no_cat} samples had no category defined!")
        if os.environ["DEBUG"] == "True":
            weights = weights[0:100]

        self.weights = weights

    def sampler(self):

        return torch.utils.data.sampler.WeightedRandomSampler(
            self.weights, len(self.weights)
        )

    def auc_based_sampler(self, auc):

        for ex, data in enumerate(self.data.iterrows()):
            classes = data.columns[np.where(data == 1)]
            n = len(classes)
            if n > 1:
                classes = classes[np.random.randint(0, n)]
            self.weights[ex] = 1 / auc[classes]

        return self.weights


if __name__ == "__main__":
    sampler = Sampler(f"{os.environ['img_dir']}/training").sampler()
