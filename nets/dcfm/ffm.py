import torch
from torch import nn
from torch.nn import functional as F


class _CONV(nn.Module):
    def __init__(self,
                 conv_layer,
                 in_channels,
                 out_channels,
                 kernel,
                 stride,
                 padding,
                 bias=True,
                 norm=True,
                 act='relu'):
        super(_CONV, self).__init__()
        self.conv = conv_layer(in_channels,
                               out_channels,
                               kernel,
                               stride,
                               padding,
                               bias=bias)
        if norm:
            self.norm = nn.SyncBatchNorm(out_channels)
        else:
            self.norm = None
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            assert 0

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, choose_chan=-1):
        super(FeatureFusionModule, self).__init__()
        self.convblk = _CONV(nn.Conv2d, in_chan, out_chan, 1, 1, 0)
        # self.conv1 = nn.Conv2d(out_chan,
        #                        out_chan // 4,
        #                        kernel_size=1,
        #                        stride=1,
        #                        padding=0,
        #                        bias=False)
        # self.conv2 = nn.Conv2d(out_chan // 4,
        #                        out_chan,
        #                        kernel_size=1,
        #                        stride=1,
        #                        padding=0,
        #                        bias=False)
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.choose_chan = choose_chan
        self.init_weight()

    def forward(self, fsp, fcp):
        ch = fcp.shape[1] if self.choose_chan == -1 else self.choose_chan
        fcat = torch.cat([fsp[:, :, :, :], fcp[:, :ch, :, :]], dim=1)
        feat = self.convblk(fcat)
        return feat
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)