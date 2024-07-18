from typing import Optional
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import sys
from torch import Tensor, nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from .video_test import video_test_Metric, VC
import math


class Metric(nn.Module):
    def __init__(self, model, **kwargs) -> None:
        super(Metric, self).__init__()
        self.model = model(**kwargs)
        self.S = 0.
        self.n = 0

    def reset(self):
        self.S = 0.
        self.n = 0

    def forward(self, output, target):
        y = self.model(output, target)
        self.S += y
        self.n += 1

    def get(self):
        return self.S / self.n


class Accuracy(nn.Module):
    def __init__(self) -> None:
        super(Accuracy, self).__init__()
        self.reset()

    def reset(self):
        self.tp = 0
        self.p = 0

    def forward(self, output, target):
        output = output.argmax(1)
        t = (output == target)
        self.tp += sum(t)
        self.p += len(target)

    def get(self):
        return self.tp / self.p


def Upsample(x, size):
    return nn.functional.interpolate(x,
                                     size=size,
                                     mode='bilinear',
                                     align_corners=False)


class mIoU(nn.Module):
    def __init__(self, C) -> None:
        super(mIoU, self).__init__()
        self.C = C
        self.reset()

    def reset(self, ):
        self.matrix = np.zeros([self.C, self.C], dtype=np.int32)

    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = Upsample(pred, target.shape[-2:])
        target = target.view(-1).cpu().numpy()
        pred = pred.argmax(1).view(-1).cpu().numpy()
        matrix = confusion_matrix(pred,
                                  target,
                                  labels=[i for i in range(self.C)])
        # for i in range(self.C):
        #     for j in range(self.C):
        #         n = int(((target == i) & (pred == j)).sum())
        #         matrix[i, j] = n
        self.matrix += matrix

    def iou(self, c):
        I = self.matrix[c, c]
        U = self.matrix[c, :].sum() + self.matrix[:, c].sum() - I
        return I / (U + 1e-15)

    def get(self, ):
        ret = 0.
        for i in range(self.C):
            ret += self.iou(i)
        return ret / self.C


class wIoU(nn.Module):
    def __init__(self, C) -> None:
        super(wIoU, self).__init__()
        self.C = C
        self.reset()

    def reset(self, ):
        self.matrix = np.zeros([self.C, self.C], dtype=np.int32)

    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = Upsample(pred, target.shape[-2:])
        target = target.view(-1).cpu().numpy()
        pred = pred.argmax(1).view(-1).cpu().numpy()
        matrix = confusion_matrix(pred,
                                  target,
                                  labels=[i for i in range(self.C)])
        self.matrix += matrix

    def get(self, ):
        f = np.sum(self.matrix, 0) / np.sum(self.matrix)
        iu = np.diag(self.matrix) / (np.sum(self.matrix, 1) +
                                     np.sum(self.matrix, 0) - np.diag(self.matrix))

        ret = (f[f > 0] * iu[f > 0]).sum()
        return ret


class macc(nn.Module):
    def __init__(self, C) -> None:
        super(macc, self).__init__()
        self.C = C
        self.reset()

    def reset(self, ):
        self.matrix = np.zeros([self.C, self.C], dtype=np.int32)

    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = Upsample(pred, target.shape[-2:])
        target = target.view(-1).cpu().numpy()
        pred = pred.argmax(1).view(-1).cpu().numpy()
        matrix = confusion_matrix(pred,
                                  target,
                                  labels=[i for i in range(self.C)])
        # for i in range(self.C):
        #     for j in range(self.C):
        #         n = int(((target == i) & (pred == j)).sum())
        #         matrix[i, j] = n
        self.matrix += matrix

    def acc(self, c):
        I = self.matrix[c, c]
        U = self.matrix[:, c].sum()
        return I / (U + 1e-15)

    def get(self, ):
        ret = 0.
        for i in range(self.C):
            ret += self.acc(i)
        return ret / self.C


class PSNR(nn.Module):
    def __init__(self) -> None:
        super(PSNR, self).__init__()

    def forward(self, output, target):
        mse = nn.MSELoss()(output, target)
        return 10 * torch.log10(1.0 * 1.0 / mse)


class test_time(nn.Module):
    def __init__(self) -> None:
        super(test_time, self).__init__()
        self.ret = 0.
        self.n = 0

    def reset(self,):
        self.ret = 0.
        self.n = 0

    def forward(self, infer_time):
        ret = 0.
        for time in infer_time:
            ret += time
        self.ret += ret / len(infer_time)
        self.n += 1

    def get(self):
        return self.ret / self.n


class CrossEntropyLoss_fix(CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, ignore_index,
                         reduce, reduction, label_smoothing)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from functools import reduce
        if (target == self.ignore_index).sum() == reduce(lambda x, y: x * y, target.shape):
            return 0.
        return super().forward(input, target)


class OhemCELoss2D(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self,
                 thresh=0.7,
                 ignore_index=-1):

        super(OhemCELoss2D, self).__init__(
            None, None, ignore_index, reduction='none')

        self.thresh = -math.log(thresh)
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return self.OhemCELoss(pred, target)

    def OhemCELoss(self, logits, labels):
        loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        n_min = len(loss) // 16
        if loss[n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)


losses = {
    "CE": CrossEntropyLoss_fix,
    "MSE": MSELoss,
    "PSNR": PSNR,
    "L1": L1Loss,
    "ohem": OhemCELoss2D,
}


def get_metric(tag: str, video_test=False, **kwargs):
    if tag in losses:
        ret = Metric(losses[tag], **kwargs)
    else:
        ret = getattr(sys.modules[__name__], tag)(**kwargs)
    if video_test:
        return video_test_Metric(model=ret)
    return ret
