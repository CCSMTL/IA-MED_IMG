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
            fpr,tpr,thresholds = roc_curve(true[:,i],pred[:,i])
            best_threshold[i]=   thresholds[np.argmax(tpr-fpr)]

        self.thresholds = best_threshold


    def convert(self,pred):

        for i in range(self.num_classes) :
            pred[:,i] = np.where(
                pred[:,i]<=self.thresholds[i],
                pred[:,i]/2/self.thresholds[i],               #if true
                1 - (1-pred[:,i])/2/(1-self.thresholds[i])    #if false
            )
        return pred

    def accuracy(self, true, pred):
        n, m = true.shape
        pred2 = self.convert(pred)
        pred2 = np.where(pred2[:, i] > 0.5, 1, 0)

        accuracy = 0
        for x, y in zip(true, pred2):
            if (x == y).all():
                accuracy += 1
        return accuracy / n

    def f1(self, true, pred):

        self.set_thresholds(true, pred)
        _, m = true.shape
        pred2 = self.convert(pred)

        pred2 = np.where(pred2 > 0.5, 1, 0)
        return metrics.f1_score(
            true, pred2, average="macro", zero_division=0
        )  # weighted??

    def precision(self, true, pred):
        _, m = true.shape
        pred2 = np.copy(pred)

        pred2 = np.where(pred2 > 0.5, 1, 0)
        return metrics.precision_score(true, pred2, average="macro", zero_division=0)

    def recall(self, true, pred):
        _, m = true.shape
        pred2 = self.convert(pred)

        pred2 = np.where(pred2 > 0.5, 1, 0)
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
        score = -np.max(pred, axis=1) + 1

        # fpr, tpr, _ = roc_curve(-np.max(true, axis=1) + 1, score)
        # outAUROC["No Finding"] = auc(fpr, tpr)
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
