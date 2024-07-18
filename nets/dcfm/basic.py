from torch import nn
import torch

from .ffm import FeatureFusionModule
from nets import hrnet
from nets.hrnet.config import config, update_config
from nets import pspnet
from nets import segformer


def get_backbone(name: str):
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
        model = segformer.segformer(name[9:], "./pretrain/mit_{}.pth".format(name[9:]))
        high_level_ch = 256
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


def stage1_name(backbone: str):
    if backbone == "hrnetv2":
        return "layer1"
    elif backbone == "hrnetv2-w18":
        return "layer1"
    elif backbone[:6] == "pspnet":
        return "layer1"
    elif backbone[:6] == "resnet":
        return "layer1"
    elif backbone[:9] == "segformer":
        return "hk"
    else:
        assert 0


class PRE(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone='hrnetv2',
                 use_aspp=False,
                 head_ch=256,
                 head_nums=1,
                 head_drop=0.,
                 **kwargs):
        super(PRE, self).__init__()
        self.backbone, high_level_ch = get_backbone(backbone)
        self.use_aspp = use_aspp
        final_in_ch = high_level_ch
        self.final = make_seg_head(in_ch=final_in_ch,
                                   out_ch=num_classes,
                                   bot_ch=head_ch,
                                   dropout=head_drop,
                                   nums=head_nums)

        initialize_weights(self.final)

    def forward(self, x):
        def norm(a):
            return torch.nn.functional.normalize(a, p=2, dim=1)

        x_size = x.size()

        features = self.backbone(x)

        final_features = features

        if self.use_aspp:
            aspp = self.aspp(final_features)
            aspp = self.bot_aspp(aspp)
        else:
            aspp = final_features
        aspp = norm(aspp)
        pred = self.final(aspp)
        pred = Upsample(pred, x_size[2:])

        return pred


class DCFM(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone='hrnetv2',
                 criterion=None,
                 freeze=False,
                 use_aspp=False,
                 bot_ch=256,
                 bk=False,
                 head_drop=0.,
                 head_ch=256,
                 head_nums=1,
                 stage1_ch=128,
                 **kwargs):
        super(DCFM, self).__init__()
        self.criterion = criterion
        self.backbone, high_level_ch = get_backbone(backbone)
        self.use_aspp = use_aspp
        self.bot_ch = bot_ch
        self.bk = bk
        if freeze:
            self.backbone.requires_grad = False

        self.bot_aspp = nn.Conv2d(high_level_ch,
                                  bot_ch,
                                  kernel_size=1,
                                  bias=False)
        self.ffm = FeatureFusionModule(bot_ch + stage1_ch, bot_ch, stage1_ch)
        self.final = make_seg_head(in_ch=bot_ch,
                                   out_ch=num_classes,
                                   nums=head_nums,
                                   dropout=head_drop,
                                   bot_ch=head_ch)

        initialize_weights(self.final)

        def hook(module, input, output):
            self.stage1_feature = output
            return None

        hook_layer_name = stage1_name(backbone=backbone)
        getattr(self.backbone, hook_layer_name).register_forward_hook(hook)

    def load(self, pretrained_dict):
        model_dict = self.state_dict()
        pretrained_dict = {
            k.replace('module.', ''): v
            for k, v in pretrained_dict.items()
        }
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict.keys()
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, inputs):
        def norm(a):
            return torch.nn.functional.normalize(a, p=2, dim=1)

        x_L, x_U = inputs
        x = torch.cat([x_U, x_L], 0)
        b = x_L.shape[0]
        x_size = x.size()

        features = self.backbone(x)
        if self.use_aspp:
            aspp = self.aspp(features)
        else:
            aspp = features
        aspp = self.bot_aspp(aspp)
        self.stage1_feature = Upsample(self.stage1_feature, aspp.shape[-2:])
        aspp = norm(aspp)

        feat_U = aspp[:b, :, :, :]
        feat_L = aspp[b:, :, :, :]
        stage1_feat_L = norm(self.stage1_feature[b:, :, :, :])
        stage1_feat_U = norm(self.stage1_feature[:b, :, :, :])

        if not self.bk:
            feat_L_hat = norm(self.ffm(feat_U, stage1_feat_L))
        else:
            feat_L_hat = norm(stage1_feat_L)
        feat_L_t = norm(self.ffm(feat_L, stage1_feat_L))
        feat_U_t = norm(self.ffm(feat_U, stage1_feat_U))

        pred_L_hat = self.final(feat_L_hat)
        pred_L_t = self.final(feat_L_t)
        B_mask = pred_L_t.argmax(1)
        pred_U_t = self.final(feat_U_t)
        I_mask = pred_U_t.argmax(1)

        inter_mask = (B_mask == I_mask).unsqueeze(1)

        pred_L_hat = Upsample(pred_L_hat, x_size[2:])
        pred_L_t = Upsample(pred_L_t, x_size[2:])
        # pred_U_t = Upsample(pred_U_t, x_size[2:])

        return pred_L_t, pred_L_hat, feat_L_t, feat_L_hat, feat_L_t * inter_mask, feat_U_t * inter_mask

        # if self.training:
        #     MSE = nn.MSELoss()
        #     assert 'gts' in inputs
        #     gts = inputs['gts']
        #     loss = {}
        #     loss["ce"] = self.lambda_B * self.criterion(
        #         pred, gts) + self.lambda_I * self.criterion(pred_t, gts)
        #     loss['difff'] = MSE(feat_B_hat, feat_B_t)
        #     loss['pc_pos'] = MSE(feat_B_t * inter_mask, feat_I_t * inter_mask)
        #     return loss
        # else:
        #     output_dict = {'pred': pred_t, 'pred_B': pred}
        #     return output_dict
