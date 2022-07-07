import torch

from torchvision import transforms


# TODO : TESTS THESE VISUALLY

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

        image1 = torch.tensor((1 - self.intensity) * image1 + self.intensity * image2, dtype=torch.uint8)
        label1 = (1 - self.intensity) * label1 + self.intensity * label2

        return image1, label1.float()

    def __call__(self, samples):
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples
        probs = torch.rand((n))
        idxs = torch.randint(0, n, (n,))  # random indexes

        mask = probs < self.prob
        idxs = idxs[mask]
        image1 = images[mask]
        image2 = images[idxs]
        label1 = labels[mask]
        label2 = labels[idxs]
        images[mask], labels[mask] = self.mixing(image1, image2, label1, label2)

        return (images, labels)


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

        bbox = torch.hstack(
            (torch.rand((n, 2)) * height, torch.abs(torch.randn((n, 2)) * height * self.intensity))).int()
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

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
        for xx, xx2, yy, yy2 in zip(x, x2, y, y2):
            image1[:, xx:xx2, yy:yy2] = image2[:, xx:xx2, yy:yy2]

        label1 = (1 - ratio[:, None]) * label1 + ratio[:, None] * label2
        return image1, label1

    def __call__(self, samples):
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples
        probs = torch.rand((n,), device=images.device)
        idxs = torch.randint(0, n, (n,), device=images.device)  # random indexes

        images[probs < self.prob], labels[probs < self.prob] = self.cutmix(images[probs < self.prob],
                                                                           images[idxs[probs < self.prob]],
                                                                           labels[probs < self.prob],
                                                                           labels[idxs[probs < self.prob]])

        return (images, labels)


class RandomErasing(object):
    """
    Class that regroups all supplementary transformations not implemented by default in pytorch aka randaugment.
    """

    def __init__(
            self,
            prob,
            intensity
    ):
        """

        :param prob: either a float or array of appropriate size
        """
        self.prob = prob
        self.intensity = intensity

    def randomerasing(self, image1, image2, label1, label2):
        n, channels, height, width = image1.shape

        bbox = torch.hstack(
            (torch.rand((n, 2)) * height, torch.abs(torch.randn((n, 2)) * height * self.intensity))).int()
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

        # lets make sure there are no out of bound boxes
        x2 = x + w
        x2 = torch.max(x2, torch.zeros_like(x2))
        x2 = torch.min(x2, torch.ones_like(x2) * height)
        y2 = y + h
        y2 = torch.max(y2, torch.zeros_like(x2))
        y2 = torch.min(x2, torch.ones_like(x2) * height)
        ratio = torch.abs(x2 - x) * torch.abs(y2 - y) / height ** 2
        x, x2, y, y2 = x.int(), x2.int(), y.int(), y2.int()

        # todo : fix this mess ....
        for xx, xx2, yy, yy2 in zip(x, x2, y, y2):
            image1[:, xx:xx2, yy:yy2] = torch.randn((n, channels, xx2 - xx, yy2 - yy))

        return image1, label1

    def __call__(self, samples):
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples
        probs = torch.rand((n,))
        idxs = torch.randint(0, n, (n,))  # random indexes

        images[probs < self.prob], labels[probs < self.prob] = self.randomerasing(images[probs < self.prob],
                                                                                  images[idxs[probs < self.prob]],
                                                                                  labels[probs < self.prob],
                                                                                  labels[idxs[probs < self.prob]])

        return (images, labels)


class RandAugment:
    def __init__(self, prob, N, M):
        self.prob = prob

        self.augment = transforms.RandAugment(num_ops=N, magnitude=M)

    def __call__(self, samples):
        if self.prob == 0:
            return samples
        images, labels = samples
        n = images.shape[0]  # number of samples
        probs = torch.rand((n))

        images[probs < self.prob] = self.augment(images[probs < self.prob])

        return (images, labels)
