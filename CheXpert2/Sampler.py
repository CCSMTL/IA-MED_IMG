<<<<<<< HEAD:CheXpert2/Sampler.py
import pandas as pd
import numpy as np
import torch
import yaml


class Sampler:
    def __init__(self, datafolder):

        with open(f"{datafolder}/data.yaml", "r") as stream:
            names = yaml.safe_load(stream)["names"]
        names += ["No Finding"]
        data = pd.read_csv(f"{datafolder}/data_table.csv")
        data = data[data["assignation"] == "training"]
        if __debug__:
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
=======
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

# calculate power curves for varying sample and effect size
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower
# parameters for power analysis
effect_sizes = array([0.2, 0.5, 0.8])
sample_sizes = array(range(5, 100))
# calculate power curves from multiple power analyses
analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()
>>>>>>> origin/zahra:Sampler.py
