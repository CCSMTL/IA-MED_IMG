import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


class Metrics:
    def __init__(self, num_classes, names, threshold=0.5):
        self.num_classes = num_classes
        self.thresholds = threshold
        self.names = names

    def set_thresholds(self, true, pred):

        best_threshold = np.zeros((self.num_classes))
        for i in range(self.num_classes):
            max_score = 0
            for threshold in np.arange(0.1, 1, 0.1):
                pred2 = np.where(np.copy(pred[:, i]) > threshold, 1, 0)
                score = metrics.f1_score(
                    true[:, i], pred2, average="macro", zero_division=0
                )  # weighted??
                if score > max_score:
                    max_score = score
                    best_threshold[i] = threshold

        self.thresholds = best_threshold
        print(self.thresholds)

    def accuracy(self, true, pred):
        n, m = true.shape
        pred2 = np.copy(pred)
        for i in range(0, m):
            pred2[:, i] = np.where(pred[:, i] > self.thresholds[i], 1, 0)

        accuracy = 0
        for x, y in zip(true, pred2):
            if (x == y).all():
                accuracy += 1
        return accuracy / n

    def f1(self, true, pred):

        self.set_thresholds(true, pred)
        _, m = true.shape
        pred2 = np.copy(pred)
        for i in range(0, m):
            pred2[:, i] = np.where(pred[:, i] > self.thresholds[i], 1, 0)
        return metrics.f1_score(
            true, pred2, average="macro", zero_division=0
        )  # weighted??

    def precision(self, true, pred):
        _, m = true.shape
        pred2 = np.copy(pred)
        for i in range(0, m):
            pred2[:, i] = np.where(pred[:, i] > self.thresholds[i], 1, 0)
        return metrics.precision_score(true, pred2, average="macro", zero_division=0)

    def recall(self, true, pred):
        _, m = true.shape
        pred2 = np.copy(pred)
        for i in range(0, m):
            pred2[:, i] = np.where(pred[:, i] > self.thresholds[i], 1, 0)
        return metrics.recall_score(true, pred2, average="macro", zero_division=0)

    def computeAUROC(self, true, pred):

        fpr = dict()
        tpr = dict()
        outAUROC = dict()
        classCount = pred.shape[1]  # TODO : add auc no finding
        for i in range(classCount):
            try:
                fpr[i], tpr[i], _ = roc_curve(true[:, i], pred[:, i])

                outAUROC[self.names[i]] = auc(fpr[i], tpr[i])
            except:
                outAUROC[self.names[i]] = 0
            if np.isnan(outAUROC[self.names[i]]):
                outAUROC[self.names[i]] = 0
        outAUROC["mean"] = np.mean(list(outAUROC.values()))
        score = -np.mean(pred, axis=1) + 1

        fpr, tpr, _ = roc_curve(-np.max(true, axis=1) + 1, score)
        outAUROC["No Finding"] = auc(fpr, tpr)
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
