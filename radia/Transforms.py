import torch

from torchvision import transforms


# TODO : TESTS THESE VISUALLY


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

    def __call__(
        self, samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image1, image2, label1, label2 = samples

        if torch.rand(1) < self.prob:
            image1 = (1 - self.intensity) * image1 + self.intensity * image2
            label1 = (1 - self.intensity) * label1 + self.intensity * label2
        return (image1, image2, label1, label2)


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

    def __call__(
        self, samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image1, image2, label1, label2 = samples
        n = image1.shape[1]
        bbox = torch.cat(
            (torch.rand(2) * n, torch.abs(torch.randn(2) * n * self.intensity))
        ).int()
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if torch.rand(1) < self.prob:
            x2 = min(x + w, n) if w > 0 else max(x + w, 0)
            y2 = min(y + h, n) if h > 0 else max(y + h, 0)

            bbox[0] = min(x, x2)
            bbox[2] = max(x, x2)
            bbox[1] = min(y, y2)
            bbox[3] = max(y, y2)

            ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / n**2

            image1[:, bbox[0] : bbox[2], bbox[1] : bbox[3]] = image2[
                :, bbox[0] : bbox[2], bbox[1] : bbox[3]
            ]
            label1 = (1 - ratio) * label1 + ratio * label2

        return (image1, image2, label1, label2)


class RandAugment:
    def __init__(self, prob, N, M):
        self.p = prob

        self.augment = transforms.RandAugment(num_ops=N, magnitude=M)

    def __call__(self, samples):
        image1, image2, label1, label2 = samples
        if torch.randn((1,)) < self.p:
            image1 = self.augment(image1)

        return (image1, image2, label1, label2)
