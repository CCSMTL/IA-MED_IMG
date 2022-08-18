import os

import numpy as np
import pandas as pd
import torch


class Sampler:
    def __init__(self, datafolder):
        no_cat = 0

        data = pd.read_csv(datafolder)
        data = data.replace(-1, 0.5).fillna(0)
        self.data = data
        count = self.data.iloc[:, 5:19].sum().to_numpy()
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

        self.weights = weights

    def sampler(self):
        if os.environ["DEBUG"] == "True":
            num_samples = 10
        else :
            num_samples = 200000
        return torch.utils.data.sampler.WeightedRandomSampler(
            self.weights,num_samples=num_samples
        )

    def auc_based_sampler(self, auc):
        names = self.data.columns[5:19]

        for ex, data in self.data.iterrows():
            classes = names[np.where(data.to_numpy()[5:19] == 1)[0]]
            n = len(classes)
            if n > 1:
                classes = classes[np.random.randint(0, n)]

            try:
                self.weights[ex] = 1 / auc[classes]
            except:
                self.weights[ex] = 10
            if n == 0:
                self.weights[ex] = 1 / auc["No Finding"]

        return self.weights


if __name__ == "__main__":
    sampler = Sampler(f"{os.environ['img_dir']}/training").sampler()
