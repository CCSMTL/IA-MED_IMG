from functools import reduce

import torch
from torch.autograd import Variable
from CheXpert2.models.CNN import CNN



class Ensemble(torch.nn.Module):
    def __init__(self, models,output_size):
        super().__init__()



        self.models = torch.nn.ModuleList(models) #list of models
        self.output_size = output_size
        self.m = len(models)
        self.eval()

    @torch.no_grad()
    def forward(self, x):

        output = torch.zeros((self.output_size))
        for model in self.models :
            output = output + model(x)
        return output/self.m




if __name__ == "__main__":  # for debugging purpose
    import time

   # start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    x = torch.zeros((1, 1, 384, 384))
    models = [
        CNN("convnext_base",img_size=384,channels=1,num_classes=14,pretrained=False,pretraining=False),
        CNN("deit3_base_patch16_384", img_size=384, channels=1, num_classes=14, pretrained=False,pretraining=False),
        CNN("densenet201", img_size=384, channels=1, num_classes=14, pretrained=False,pretraining=False),
    ]
    ensemble = Ensemble(models,14)
    #ensemble = ensemble.cuda()
    #start.record()
    import time
    start= time.time()
    output = ensemble(x)
    #end.record()
    end=time.time()
    #torch.cuda.synchronize()
    # print("time : ", start.elapsed_time(end))
    print(end-start)
    print(output)
