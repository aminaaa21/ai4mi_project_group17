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


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


# ADDED Cross-entropy + soft Dice loss (main.py --loss ce_dice, used in hu_window/run_dice_loss.sh)
#
# The cross-entropy is an average over all the pixels, so a tiny organ barely counts in it
# (the trachea is about 0.04% of the pixels). The soft Dice is computed per class and then
# averaged, so every organ counts the same, whatever its size. It is computed on the
# probabilities (not on the argmax like the Dice metric), so it already rewards the network when
# the probability of an organ goes up, before that organ is actually predicted.
class CrossEntropyDice():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']  # Classes supervised by the cross-entropy
        self.dice_idk = kwargs['dice_idk']  # Classes in the Dice term (the organs, not the background)
        self.cross_entropy = CrossEntropy(idk=self.idk)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        ce_loss = self.cross_entropy(pred_softmax, weak_target)

        probs = pred_softmax[:, self.dice_idk, ...]
        target = weak_target[:, self.dice_idk, ...].float()

        # One soft Dice per class, summed over the whole batch (b, w, h) instead of per slice:
        # many slices do not contain every organ, which would make a per-slice Dice unstable.
        intersection = einsum("bkwh,bkwh->k", probs, target)
        sizes = einsum("bkwh->k", probs) + einsum("bkwh->k", target)
        # The +1 avoids a division by 0 when an organ is in neither the batch nor the prediction
        dice = (2 * intersection + 1) / (sizes + 1)
        dice_loss = 1 - dice.mean()

        return ce_loss + dice_loss
