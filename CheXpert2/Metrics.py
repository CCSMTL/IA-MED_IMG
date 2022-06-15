import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import warnings
import yaml
import sys

with open("data/data.yaml", "r") as stream:  # TODO : remove hardcode
    names = yaml.safe_load(stream)["names"]

names += ["No Finding"]


class Metrics:
    def __init__(self, num_classes, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold

    def accuracy(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return np.mean(np.where(pred == true, 1, 0))

    def f1(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.f1_score(
            true, pred, average="macro", zero_division=0
        )  # weighted??

    def precision(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.precision_score(true, pred, average="macro", zero_division=0)

    def recall(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.recall_score(true, pred, average="macro", zero_division=0)

    def computeAUROC(self, true, pred):
        try:

            fpr = dict()
            tpr = dict()
            outAUROC = dict()
            classCount = pred.shape[1]
            for i in range(classCount):
                fpr[i], tpr[i], _ = roc_curve(true[:, i], pred[:, i])
                outAUROC[names[i]] = auc(fpr[i], tpr[i])
            outAUROC["mean"] = np.mean(list(outAUROC.values()))
        except ValueError as e:
            print(e, file=sys.stderr)
            for i in names + ["mean"]:
                outAUROC[i] = 0  # TODO : set to something more meaningfull? aka -1?
        return outAUROC

    def metrics(self):
        dict = {
            "f1": self.f1,
            "auc": self.computeAUROC,
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
        }
        return dict
