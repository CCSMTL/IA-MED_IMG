import torch
import numpy as np

class Mixing(object):
    """
    Class that regroups all supplementary transformations not implemented by default in pytorch aka randaugment.
    """

    def __init__(self, prob,intensity):
        """

        :param prob: either a float or array of appropriate size
        """
        self.prob=prob
        self.intensity=intensity
    def __call__(self,samples):
        image1, landmarks1 = samples['image'], samples['landmarks']
        image2, landmarks2 = samples['image2'], samples['landmarks2']

        if torch.rand(1)<self.prob :
            samples['image']=(1-self.intensity)*image1+self.intensity*image2
            samples['landmarks']=(1-self.intensity)*landmarks1+self.intensity*landmarks2
        return samples


class CutMix(object):
    """
    Class that regroups all supplementary transformations not implemented by default in pytorch aka randaugment.
    """

    def __init__(self, prob):
        """

        :param prob: either a float or array of appropriate size
        """
        self.prob = prob


    def __call__(self, samples):
        image1, landmarks1 = samples['image'], samples['landmarks']
        image2, landmarks2 = samples['image2'], samples['landmarks2']
        n=image1.shape[1]
        bbox=torch.cat((torch.rand(2)*n,torch.randn(2)*n/5)).int()
        x,y,w,h=bbox
        ratio=w*h/n**2
        #TODO : make sure bbox!=0
        if torch.rand(1) < self.prob:
            image1[:,x:x+w,y:y+h]=image2[:,x:x+w,y:y+h]
            landmarks1 = (1 - ratio) * landmarks1 + ratio * landmarks2

        samples['image'], samples['landmarks']=image1,landmarks1
        return samples



class RandomErasing(object):
    """
    Class that regroups all supplementary transformations not implemented by default in pytorch aka randaugment.
    """

    def __init__(self, prob,):
        """

        :param prob: either a float or array of appropriate size
        """
        self.prob = prob


    def __call__(self, samples):
        image1, landmarks1 = samples['image'], samples['landmarks']

        m,n,_=image1.shape

        bbox=torch.cat((torch.rand(2)*n,torch.abs(torch.randn(2)*n/5))).int()
        x,y,w,h=bbox


        if torch.rand(1) < self.prob:
            x2 = min(x+w,n) if w>0 else max(x+w,0)
            y2 = min(y+h,n) if h>0 else max(y+h,0)
            if x2 < x :
                x, x2 = x2, x
            if y2 < y :
                y, y2 = y2, y

            image1[:,x:x2,y:y2]=torch.randint(0,255,(m,x2-x,y2-y))

        samples['image'], samples['landmarks']=image1,landmarks1
        return samples


