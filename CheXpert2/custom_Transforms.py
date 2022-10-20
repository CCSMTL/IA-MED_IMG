import torch

from torchvision import transforms


# TODO : TESTS THESE VISUALLY

@torch.jit.script
def invert_smaller(x1: int, x2: int) -> tuple[int, int]:
    if x1 > x2:
        x1,x2 =  x2,x1
    return (x1, x2)


@torch.no_grad()
@torch.jit.script
class Mixing(object):
    """
    Class that regroups all supplementary transformations not implemented by default in pytorch aka randaugment.
    """

    def __init__(self, prob, intensity):
        """

        :param prob: either a float or array of appropriate size
        """
        self.prob = prob
        self.intensity = intensity

    def mixing(self, image1, image2, label1, label2):
        """
        This function is used to mix two images.
        :param image:
        :param label:
        :return: mixed image and labels
        """

        image1 = ((1 - self.intensity) * image1 + self.intensity * image2).byte()
        label1 = (1 - self.intensity) * label1 + self.intensity * label2

        return image1, label1

    def __call__(self, samples: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples
        probs = torch.rand((n))

        idx1 = probs < self.prob

        image1 = images[idx1]
        label1 = labels[idx1]
        m = len(image1)
        idx2 = torch.randperm(n)[:m]
        image2 = images[idx2]

        label2 = labels[idx2]
        images[idx1], labels[idx1] = self.mixing(image1, image2, label1, label2)

        return (images, labels)


@torch.no_grad()
@torch.jit.script
class CutMix(object):
    """
    Class that regroups all supplementary transformations not implemented by default in pytorch aka randaugment.
    """

    def __init__(self, prob, intensity):
        """

        :param prob: either a float or array of appropriate size
        """
        self.prob = prob
        self.intensity = intensity

    def cutmix(self, image1, image2, label1, label2):

        n, _, height, _ = image1.shape

        x = torch.rand((n,)) * height
        y = torch.rand((n,)) * height
        h = torch.randn((n,)) * height * self.intensity
        w = torch.randn((n,)) * height * self.intensity

        # lets make sure there are no out of bound boxes
        x2 = x + w
        x2 = torch.max(x2, torch.zeros_like(x2))
        x2 = torch.min(x2, torch.ones_like(x2) * height)
        y2 = y + h
        y2 = torch.max(y2, torch.zeros_like(x2))
        y2 = torch.min(y2, torch.ones_like(x2) * height)

        ratio = torch.abs(x2 - x) * torch.abs(y2 - y) / height ** 2
        ratio = ratio.to(image1.device)
        x, x2, y, y2 = x.int(), x2.int(), y.int(), y2.int()
        i = 0
        for xx, xx2, yy, yy2 in zip(x, x2, y, y2):
            xx, xx2 = invert_smaller(xx, xx2)
            yy, yy2 = invert_smaller(yy, yy2)

            # TODO : vectorize this loop
            image1[i, :, xx:xx2, yy:yy2] = image2[i, :, xx:xx2, yy:yy2]
            i += 1

        label1 = (1 - ratio[:, None]) * label1 + ratio[:, None] * label2

        return image1, label1

    def __call__(self, samples: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples

        probs = torch.rand((n,), device=images.device)
        idx1 = probs < self.prob

        image1 = images[idx1]
        label1 = labels[idx1]
        m = len(image1)
        idx2 = torch.randperm(n)[:m]
        image2 = images[idx2]

        label2 = labels[idx2]
        images[idx1], labels[idx1] = self.cutmix(image1, image2, label1, label2)
        return (images, labels)


@torch.no_grad()
class RandAugment:
    def __init__(self, prob, N, M):
        self.prob = prob

        self.augment = transforms.RandAugment(num_ops=N, magnitude=M)

    def __call__(self, samples: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples
        probs = torch.rand((n))
        if sum(probs < self.prob).item() > 0:
            images[probs < self.prob, :, :, :] = self.augment(images[probs < self.prob, :, :, :])

        return (images, labels)
