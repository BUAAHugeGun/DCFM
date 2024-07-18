from torch import nn
import numpy as np


class video_test_Metric(nn.Module):
    def __init__(self, model) -> None:
        super(video_test_Metric, self).__init__()
        self.model = model
        self.model.reset()

    def reset(self):
        self.model.reset()

    def forward(self, preds, gt):
        for id, target in gt.items():
            if preds[id] is not None:
                self.model(preds[id], target)

    def get(self):
        return self.model.get()


class VC(nn.Module):
    def __init__(self, n) -> None:
        super(VC, self).__init__()
        self.n = n
        self.reset()

    def reset(self, ):
        self.accs = []

    def forward(self, preds, gt):
        assert len(preds) == len(gt)
        h, w = preds[0].shape[-2:]
        length = len(preds)
        np_preds = [x.argmax(1).view(-1).cpu().numpy() for x in preds]
        np_gt = [gt[i].view(-1).cpu().numpy() for i in range(length)]

        video_acc = []
        for i in range(length - self.n):
            global_common = np.ones((h * w))
            predglobal_common = np.ones((h * w))
            for j in range(1, self.n):
                common = (np_gt[i] == np_gt[i + j])
                global_common = np.logical_and(global_common, common)
                pred_common = (np_preds[i] == np_preds[i + j])
                predglobal_common = np.logical_and(
                    predglobal_common, pred_common)
            pred = (predglobal_common * global_common)
            acc = pred.sum() / (global_common.sum() + 1e-8)
            video_acc.append(acc)
        self.accs.extend(video_acc)

    def get(self, ):
        return sum(self.accs) / len(self.accs)
