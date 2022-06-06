import numpy as np
from sklearn import metrics
from sklearn.metrics._ranking import roc_auc_score
import warnings
import yaml

with open("data/data.yaml", "r") as stream:
    names = yaml.safe_load(stream)["names"]

names += ["No Finding"]


class Metrics:
    def __init__(self, num_classes, threshold):
        self.num_classes = num_classes
        self.threshold = threshold

    def accuracy(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return np.mean(np.where(pred == true, 1, 0))

    def f1(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.f1_score(true, pred, average="macro",zero_division=0)  # weighted??

    def precision(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.precision_score(true, pred, average="macro",zero_division=0)

    def recall(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.recall_score(true, pred, average="macro",zero_division=0)

    def computeAUROC(self, true, pred):
        try:
            outAUROC = {}
            classCount = pred.shape[1]
            for i in range(classCount):
                outAUROC[names[i]] = roc_auc_score(true[:, i], pred[:, i])
            outAUROC["mean"] = np.mean(list(outAUROC.values()))
        except ValueError as e:
            warnings.warn(str(e))
            for i in names + ["mean"]:
                outAUROC[i] = 0
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
