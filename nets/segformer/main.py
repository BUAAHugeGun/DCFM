import torch
from torch import nn
from .mix_transformer import mit_b2, mit_b1, mit_b5, mit_b3, mit_b4, mit_b0
from .segformer_head import SegFormerHead


class segformer(nn.Module):
    def __init__(self, mode, pretrain, embed_dim):
        super(segformer, self).__init__()
        if mode == "b2":
            self.encoder = mit_b2()
        elif mode == "b1":
            self.encoder = mit_b1()
        elif mode == "b0":
            self.encoder = mit_b0()
        elif mode == "b3":
            self.encoder = mit_b3()
        elif mode == "b4":
            self.encoder = mit_b4()
        elif mode == "b5":
            self.encoder = mit_b5()

        if mode == "b0":
            in_channels = [32, 64, 160, 256]
        else:
            in_channels = [64, 128, 320, 512]

        self.decoder = SegFormerHead(in_channels=in_channels,
                                     in_index=[0, 1, 2, 3],
                                     feature_strides=[4, 8, 16, 32],
                                     channels=128,
                                     dropout_ratio=0.1,
                                     num_classes=19,
                                     norm_cfg=dict(
                                         type='SyncBN', requires_grad=True),
                                     align_corners=False,
                                     dim=embed_dim)

        self.hk = self.encoder.hk
        self.hk1 = self.encoder.hk1
        self.hk0 = self.encoder.hk0

        pretrained_dict = torch.load(pretrain, map_location={'cuda:0': 'cpu'})
        model_dict = self.encoder.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict.keys()
        }
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)

    def forward(self, x):
        return self.decoder(self.encoder(x))
