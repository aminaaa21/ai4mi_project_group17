#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class DiceLoss():

    def __init__(self, idk):
        self.idk = idk

    def __call__(self, pred_softmax, target):
        # Only use the selected classes
        pred = pred_softmax[:, self.idk, ...]
        target = target[:, self.idk, ...].float()

        # Calculate overlap between prediction and target
        intersection = (pred * target).sum(dim=(0, 2, 3))

        # Calculate the total size of prediction and target
        total = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))

        # Calculate Dice score
        dice = (2 * intersection + 1e-10) / (total + 1e-10)

        # Turn Dice score into a loss
        loss = 1 - dice.mean()

        return loss

class WeightedCrossEntropy():
    def __init__(self, idk, weights):
        self.idk = idk
        self.weights = weights

    def __call__(self, pred_softmax, target):
        # Only use the selected classes
        pred = pred_softmax[:, self.idk, ...]
        target = target[:, self.idk, ...].float()

        # Log of the predicted probabilities
        log_p = (pred + 1e-10).log()

        # Turn the class weights into a tensor
        weights = torch.tensor(
            self.weights,
            dtype=pred.dtype,
            device=pred.device
        )

        # Reshape so the weights can be applied to every pixel
        weights = weights.view(1, -1, 1, 1)

        # Give each class its own weight
        weighted_target = target * weights

        # Calculate weighted cross-entropy
        loss = -einsum("bkwh,bkwh->", weighted_target, log_p)

        # Average the loss
        loss /= weighted_target.sum() + 1e-10

        return loss

class DiceWeightedCELoss():
    def __init__(self, weights):
        # Dice only for foreground classes
        self.dice_loss = DiceLoss(idk=[1, 2, 3, 4])

        # Weighted CE for all five classes
        self.ce_loss = WeightedCrossEntropy(
            idk=[0, 1, 2, 3, 4],
            weights=weights
        )

    def __call__(self, pred_softmax, target):
        dice = self.dice_loss(pred_softmax, target)
        ce = self.ce_loss(pred_softmax, target)

        loss = dice + ce
        return loss
    
class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
