from torch.optim.lr_scheduler import _LRScheduler
import random
import numpy as np
import torch
from torch.backends import cudnn
import os
import yaml
from .to_log import to_log


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_ckpt(models, epoch, root, rm_ddp=False):

    def _detect_latest(name):
        checkpoints = os.listdir(os.path.join(root, "logs"))
        checkpoints = [
            f for f in checkpoints
            if f.startswith("{}_epoch_".format(name)) and f.endswith(".pth")
        ]
        checkpoints = [
            int(f[len("{}_epoch_".format(name)):-len(".pth")]) for f in checkpoints
        ]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    if epoch is None:
        return -1
    for name, model in models.items():
        if epoch == -1:
            epoch = _detect_latest(name)
        pth_path = os.path.join(root,
                                "logs/" + name + "_epoch_{}.pth".format(epoch))
        if not os.path.exists(pth_path):
            print("can't find pth file: {}".format(name))
            continue
        ckpt = torch.load(pth_path, map_location="cpu")
        if rm_ddp:
            ckpt = {k[7:]: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from iter: {}".format(name, epoch))
    return epoch


class PolyScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 power=1.0,
                 T_max=None,
                 eta_min=0,
                 last_epoch=-1,
                 warm_iter=1500,
                 warm_ratio=1e-6,
                 verbose=False):

        self.optimizer = optimizer
        self.min_lr = eta_min
        self.power = power
        self.warm_iter = warm_iter
        self.warm_ratio = warm_ratio
        self.total_steps = T_max
        super(PolyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step_num = self.last_epoch
        if step_num < self.warm_iter:
            t = (1 - self.warm_ratio) * step_num / \
                self.warm_iter + self.warm_ratio
            return [base_lr * t for base_lr in self.base_lrs]

        coeff = (1 - (step_num - self.warm_iter) /
                 (self.total_steps - self.warm_iter)) ** self.power

        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]
