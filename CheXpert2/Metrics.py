import numpy as np
import timm.utils.metrics
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


class Metrics:
    def __init__(self, num_classes, names, threshold):
        self.num_classes = num_classes
        self.thresholds = threshold
        self.names = names

    def convert(self,pred):

        for i in range(self.num_classes) :
            pred[:,i] = np.where(
                pred[:,i]<=self.thresholds[i],
                pred[:,i]/2/self.thresholds[i],               #if true
                1 - (1-pred[:,i])/2/(1-self.thresholds[i])    #if false
            )
        return pred

    def accuracy(self, true, pred):
        # n, m = true.shape
        # pred2 = self.convert(pred)
        # pred2 = np.where(pred2 > 0.5, 1, 0)
        #
        # accuracy = 0
        # for x, y in zip(true, pred2):
        #     if (x == y).all():
        #         accuracy += 1
        # accuracy /=n
        accuracy = timm.utils.metrics.accuracy(pred,true, topk=(1,))
        return accuracy

    def accuracy3(self, true, pred):
        # n, m = true.shape
        # pred2 = self.convert(pred)
        # pred2 = np.where(pred2 > 0.5, 1, 0)
        #
        # accuracy = 0
        # for x, y in zip(true, pred2):
        #     if (x == y).all():
        #         accuracy += 1
        # accuracy /=n
        accuracy = timm.utils.metrics.accuracy(pred,true, topk=(3,))
        return accuracy
    def accuracy(self, true, pred):
        pred = self.convert(pred)
        pred = np.where(pred > 0.5, 1, 0)

        accuracy = 0
        for x, y in zip(true, pred):
            if (x == y).all():
                accuracy += 1
        #accuracy = timm.utils.metrics.accuracy(pred,true, topk=(1,))
        return accuracy

    def f1(self, true, pred):

        pred2 = self.convert(pred)

        pred2 = np.where(pred2 > 0.5, 1, 0)
        return metrics.f1_score(
            true, pred2, average="weighted", zero_division=0
        )  # weighted??

    def precision(self, true, pred):

        pred = self.convert(pred)
        pred = np.where(pred > 0.5, 1, 0)
        results = metrics.precision_score(true, pred, average=None, zero_division=0)

        results_dict = {}
        for item, name in zip(results, self.names):
            results_dict[name] = item
        return results_dict

    def recall(self, true, pred):

        pred = self.convert(pred)
        pred = np.where(pred > 0.5, 1, 0)
        results=metrics.recall_score(true, pred, average=None, zero_division=0)
        results_dict={}
        for item,name in zip(results,self.names) :
            results_dict[name] = item
        return results_dict

    def computeAUROC(self, true, pred):
        print(true.shape,pred.shape)
        fpr = dict()
        tpr = dict()
        outAUROC = dict()
        classCount = pred.shape[1]  # TODO : add auc no finding
        for i in range(classCount):

            fpr[i], tpr[i], thresholds = roc_curve(true[:, i], pred[:, i])

            threshold = thresholds[np.argmax(tpr[i] - fpr[i])]
            print(f"threshold {self.names[i]} : ",threshold)
            self.thresholds[i] =threshold

            outAUROC[self.names[i]] = auc(fpr[i], tpr[i])
            if np.isnan(outAUROC[self.names[i]]):
                outAUROC[self.names[i]] = 0

        outAUROC["mean"] = np.mean(list(outAUROC.values()))


        return outAUROC

    def metrics(self):
        dict = {
            "auc": self.computeAUROC,
            "f1": self.f1,
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
        #    "accuracy-3" : self.accuracy3
        }
        return dict
