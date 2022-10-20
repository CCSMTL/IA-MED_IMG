import functools

import torch
from torch.autograd import Variable
from CheXpert2.custom_utils import channels321,Identity
import copy
from CheXpert2.models.CNN import CNN
from CheXpert2.models.Ensemble import Ensemble
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM,FullGrad

class student(torch.nn.Module) :
    def __init__(self,student_models,teacher_model,num_classes,img_size,channels,drop_rate,global_pool):
       super().__init__()


       self.teacher = Ensemble(student_models)

       self.student=torch.hub.load('ultralytics/yolov5', "_create",f'{teacher_model}.pt',channels=1)

       self.teacher_cam = FullGrad(self.teacher)





    def convert(self,outputs):#convert the outputs using the optimal threshold
        pass

    def heatmap_to_bbox(self,heatmap):
        pass


    def forward(self,inputs):
        outputs=student(inputs)
        idxs=torch.where(outputs>0.5)
        for idx in idxs :
            heatmap = self.teacher_cam(inputs,targets=[idx])
            bbox = self.heatmap_to_bbox(heatmap)

        classes = self.classifier(x+y)



if __name__ == "__main__":  # for debugging purpose
    x = torch.zeros((2, 1, 320, 320))
    for name in ["densenet121", "resnet18"]:
        cnn = CNN(name, 14)
        y = cnn(x)  # test forward loop
