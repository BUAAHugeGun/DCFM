from torch import nn
import torch

from nets.dcfm.ffm import FeatureFusionModule
from nets import hrnet
from nets.hrnet.config import config, update_config
from nets import pspnet
from nets import segformer


def get_backbone(name: str, **kwargs):
    high_level_ch = 0
    if name == "hrnetv2":
        path = "./nets/hrnet/config/w48.yaml"
        update_config(config, path=path)
        model = hrnet.seg_hrnet.get_seg_model(config)
        model.init_weights("./pretrain/hrnetv2_w48_imagenet_pretrained.pth")
        high_level_ch = 720
    elif name == "hrnetv2-w18":
        path = "./nets/hrnet/config/w18.yaml"
        update_config(config, path=path)
        model = hrnet.seg_hrnet.get_seg_model(config)
        model.init_weights("./pretrain/hrnet_w18_small_model_v2.pth")
        high_level_ch = 270
    elif name[:6] == "pspnet":
        model = pspnet.PSPNet(layers=int(name[6:]))
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        high_level_ch = 4096
    elif name[:6] == "resnet":
        model = pspnet.PSPNet(layers=int(name[6:]), use_ppm=False)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        high_level_ch = 2048
    elif name[:9] == "segformer":
        high_level_ch = kwargs.get("high_level_ch", 256)
        model = segformer.segformer(
            name[9:], "./pretrain/mit_{}.pth".format(name[9:]), embed_dim=high_level_ch)
    else:
        assert 0

    return model, high_level_ch


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.SyncBatchNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def Upsample(x, size):
    return nn.functional.interpolate(x,
                                     size=size,
                                     mode='bilinear',
                                     align_corners=False)


def make_seg_head(in_ch, out_ch, bot_ch=256, nums=2, dropout=0.):
    if nums == 2:
        layers = [
            nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(bot_ch, momentum=0.1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0001:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False))
        return nn.Sequential(*layers)
    elif nums == 1:
        layers = [
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.SyncBatchNorm(in_ch, momentum=0.1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0001:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        return nn.Sequential(*layers)

    else:
        assert 0


def stage1_name(backbone: str, stage1:str):
    if backbone == "hrnetv2":
        return "layer1"
    elif backbone == "hrnetv2-w18":
        return "layer1"
    elif backbone[:6] == "pspnet":
        return "layer1"
    elif backbone[:6] == "resnet":
        return "layer1"
    elif backbone[:9] == "segformer":
        if stage1 == "vit3":
            return "hk"
        elif stage1 == "vit1":
            return "hk1"
        elif stage1 == "vit0":
            return "hk0"
    else:
        assert 0


class PREEncoder(nn.Module):
    def __init__(self, backbone='hrnetv2', **kwargs):
        super(PREEncoder, self).__init__()
        self.backbone, high_level_ch = get_backbone(backbone, **kwargs)

    def forward(self, x):
        features = self.backbone(x)
        return features


class DCFMEncoder(nn.Module):
    def __init__(self,
                 backbone='hrnetv2',
                 freeze=False,
                 stage1="vit3",
                 **kwargs):
        super(DCFMEncoder, self).__init__()
        self.backbone, high_level_ch = get_backbone(backbone, **kwargs)
        if freeze:
            self.backbone.requires_grad = False

        def hook(module, input, output):
            self.stage1_feature = output
            return None

        hook_layer_name = stage1_name(backbone=backbone, stage1=stage1)
        getattr(self.backbone, hook_layer_name).register_forward_hook(hook)

    def forward(self, inputs):
        x_L, x_U = inputs
        x = torch.cat([x_U, x_L], 0)
        features = self.backbone(x)
        return features, self.stage1_feature


class DCFMEncoder_flops(nn.Module):
    def __init__(self,
                 backbone='hrnetv2',
                 freeze=False,
                 stage1="vit3",
                 **kwargs):
        super(DCFMEncoder_flops, self).__init__()
        self.backbone, high_level_ch = get_backbone(backbone, **kwargs)
        if freeze:
            self.backbone.requires_grad = False

        def hook(module, input, output):
            self.stage1_feature = output
            return None

        hook_layer_name = stage1_name(backbone=backbone, stage1=stage1)
        getattr(self.backbone, hook_layer_name).register_forward_hook(hook)

    def forward(self, x):
        features = self.backbone(x)
        return features, self.stage1_feature

if __name__ == "__main__":
    model = DCFMEncoder_flops(backbone="segformerb5", high_level_ch=256).eval().cuda()
    inputs = torch.randn([1,3,480, 853]).cuda()
    from thop import profile
    flops, params = profile(model, (inputs,))
    print(flops,params)
    f1, fs = model(inputs)
    print(f1.shape, fs.shape)