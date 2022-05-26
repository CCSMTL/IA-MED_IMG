import numpy as np
from sklearn import metrics


class Metrics:
    def __init__(self, num_classes, threshold):
        self.num_classes = num_classes
        self.threshold = threshold

        self.f1_list = np.zeros((num_classes))
        self.mvg_avg = 0.9

    def accuracy(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return np.mean(np.where(pred == true, 1, 0))

    def f1(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.f1_score(true, pred, average="macro")  # weighted??

    def precision(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.precision_score(true, pred, average="macro")

    def recall(self, true, pred):
        pred = np.where(pred > self.threshold, 1, 0)
        return metrics.recall_score(true, pred, average="macro")

    def auc(self, true, pred):
        # TODO :  fix auc for multiclass
        true, pred = true.T, pred.T
        auc = 0
        n = len(true)
        tpr_list, fpr_list = [], []
        cat = 0
        for t, p in zip(true, pred):  # for each class

            range_list = np.arange(0, 1.01, 0.01)
            for ex, threshold in enumerate(range_list):
                p = np.where(p > threshold, 1, 0)
                tpr = np.mean(np.where(np.logical_and(t == p, t == 1), 1, 0))
                tnr = np.mean(np.where(np.logical_and(t == p, t == 0), 1, 0))
                fnr = np.mean(np.where(np.logical_and(t != p, t == 0), 1, 0))
                fpr = np.mean(np.where(np.logical_and(t != p, t == 1), 1, 0))
                fpr_list.append(fpr)
                tpr_list.append(tpr)

                f1 = tpr / (tpr + 0.5 * (fpr + fnr))
                if f1 > self.f1_list[cat]:
                    self.threshold[cat] = (
                        self.mvg_avg * self.threshold[cat]
                        + (1 - self.mvg_avg) * threshold
                    )
                    self.f1_list[cat] = f1

            auc += np.trapz(tpr_list, fpr_list)
            cat += 1

        return auc / n

    def metrics(self):
        dict = {
            "f1": self.f1,
            #    "auc": self.auc,
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
        }
        return dict
