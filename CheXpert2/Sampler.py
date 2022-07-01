import os

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml


class Sampler:
    def __init__(self, datafolder):

        if os.path.exists(f"{datafolder}/sampler_weights.txt"):
            weights = np.loadtxt(f"{datafolder}/sampler_weights.txt")
        else:
            category_ids = []
            print("Counting files for weights calculations")
            for file in tqdm.tqdm(os.listdir(f"{datafolder}/labels")) :
                with open(f"{datafolder}/labels/{file}") as f :
                    lines = f.readlines()
                    ll = len(lines)
                    ll = np.random.randint(0, ll)
                    for line in lines:
                        line = line.split(" ")
                        category_ids.append(int(line[ll]))
                        break

            count={}
            items=np.unique(category_ids)
            for item in items :
                count[item]=np.sum(np.where(category_ids==item,1,0))
            weights=np.zeros_like(category_ids)
            for item in items :
                weights[np.where(category_ids==item)]=count[item]

            np.savetxt(f"{datafolder}/sampler_weights.txt",weights)


        if os.environ["DEBUG"] == "True":
            weights = weights[0:100]
        self.weights=weights

    def sampler(self):

        return torch.utils.data.sampler.WeightedRandomSampler(
            self.weights, len(self.weights)
        )


if __name__=="__main__" :

    sampler=Sampler(f"{os.environ['img_dir']}/training").sampler()