from torchvision import transforms
import os
import torch
import pathlib
import sklearn
import numpy as np
from  sklearn.metrics import top_k_accuracy_score
import wandb
#-----------------------------------------------------------------------------------
class Experiment() :
    def __init__(self,directory,is_wandb=False,tags=[]):
        self.is_wandb=is_wandb
        self.directory="log/"+directory
        self.weight_dir = "models/models_weights/" + directory
        for tag in tags :
            self.directory  += f"/{tag}"
            self.weight_dir += f"/{tag}"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        path=pathlib.Path(self.weight_dir)
        path.mkdir(parents=True,exist_ok=True)

        root,dir,files = list(os.walk(self.directory))[0]
        for f in files:
            os.remove(root+"/"+f)



    def log_metric(self,metric_name,value,epoch):

        f=open(f"{self.directory}/{metric_name}.txt","a")
        if type(value)==list :
            f.write("\n".join(str(item) for item in value))
        else :
            f.write(f"{epoch} , {str(value)}")

        if self.is_wandb :
            wandb.log({metric_name: value})
    def save_weights(self,model):

        torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")

        if self.is_wandb :
            wandb.save(f"{self.weight_dir}/{model._get_name()}.pt")

#-----------------------------------------------------------------------------------
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
#-----------------------------------------------------------------------------------
class preprocessing() :
    def __init__(self,img_size,other=None):
        self.img_size=img_size
        self.added_transform=other

    def preprocessing(self):
        temp=[
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),

        ]
        if self.added_transform :
            for transform in self.added_transform :
                temp.append(transform)
        temp=temp + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
        preprocess = transforms.Compose(temp)
        return preprocess
#-----------------------------------------------------------------------------------
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

#-----------------------------------------------------------------------------------
num_classes = 14 #+empty



class metrics :
    def __init__(self,num_classes,threshold):
        self.num_classes=num_classes
        self.threshold=threshold

        self.f1_list=np.zeros((14))
        self.mvg_avg=0.99


    def accuracy(self,true, pred):
        pred=np.where(pred>self.threshold,1,0)



        return np.mean(np.where(pred==true,1,0))


    def f1(self,true, pred):
        pred=np.where(pred>self.threshold,1,0)

        return sklearn.metrics.f1_score(true, pred, average='macro')  # weighted??

    def precision(self,true, pred):
        pred = np.where(pred > self.threshold,1,0)
        return sklearn.metrics.precision_score(true, pred, average='macro')

    def recall(self,true, pred):
        pred = np.where(pred > self.threshold,1,0)
        return sklearn.metrics.recall_score(true, pred, average='macro')

    def auc(self,true,pred):
        #TODO :  implement auc
        true,pred=true.T,pred.T
        auc=0
        n=len(true)
        tpr_list,fpr_list=[],[]
        cat=0
        for t,p in zip(true,pred) : #for each class
            best_auc=0
            range_list=np.arange(0,1.01,0.01)
            for ex,threshold in enumerate(range_list) :
                p=np.where(p>threshold,1,0)
                tpr=np.mean(np.where(np.logical_and(t==p,t==1),1,0))
                tnr=np.mean(np.where(np.logical_and(t==p,t==0),1,0))
                fnr = np.mean(np.where(np.logical_and(t != p, t == 0), 1, 0))
                fpr = np.mean(np.where(np.logical_and(t != p, t == 1), 1, 0))
                fpr_list.append(fpr)
                tpr_list.append(tpr)

                f1=tpr/(tpr+0.5*(fpr+fnr))
                if f1>self.f1_list[cat] :
                    self.threshold[cat]=self.mvg_avg*self.threshold[cat]+(1-self.mvg_avg)*threshold
                    self.f1_list[cat]=f1

            auc+=np.trapz(tpr_list,fpr_list)
            cat+=1

        return auc/n

    def metrics(self):
        dict={
            "f1" : self.f1,
            "auc" : self.auc,
            "recall" : self.recall,
            "precision" : self.precision,
            "accuracy" : self.accuracy
        }
        return dict
