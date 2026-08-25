"""Minimal image+target transforms for torchvision detection models
(ToTensor + optional random horizontal flip that also flips the boxes)."""

import random

import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width = F.get_image_size(image)[0]
            image = F.hflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def get_transform(train: bool):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
