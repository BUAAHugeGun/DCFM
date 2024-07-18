from torch import nn
import torch

from nets.dcfm.ffm import FeatureFusionModule


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
    elif nums == 0:
        layers = []
        if dropout > 0.0001:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        return nn.Sequential(*layers)
    else:
        assert 0


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


class PREDecoder(nn.Module):
    def __init__(self,
                 num_classes,
                 head_ch=256,
                 head_nums=1,
                 head_drop=0.,
                 high_level_ch=720,
                 **kwargs):
        super(PREDecoder, self).__init__()
        final_in_ch = high_level_ch
        self.final = make_seg_head(in_ch=final_in_ch,
                                   out_ch=num_classes,
                                   bot_ch=head_ch,
                                   dropout=head_drop,
                                   nums=head_nums)
        initialize_weights(self.final)

    def forward(self, inputs):
        def norm(a):
            return torch.nn.functional.normalize(a, p=2, dim=1)

        features, gt = inputs
        x_size = gt.shape

        features = norm(features)
        pred = self.final(features)
        pred = Upsample(pred, x_size[-2:])

        return pred


class DCFMDecoder(nn.Module):
    def __init__(self,
                 num_classes,
                 bot_ch=256,
                 head_drop=0.,
                 head_ch=256,
                 head_nums=1,
                 stage1_ch=128,
                 high_level_ch=720,
                 kh=True,
                 kl=True,
                 nh=True,
                 nl=True,
                 **kwargs):
        super(DCFMDecoder, self).__init__()
        self.bot_ch = bot_ch
        self.kh, self.kl, self.nh, self.nl = kh, kl, nh, nl

        if bot_ch > 0:
            self.bot_aspp = nn.Conv2d(high_level_ch,
                                    bot_ch,
                                    kernel_size=1,
                                    bias=False)
        else:
            self.bot_aspp = nn.Identity()
            bot_ch = high_level_ch
        self.ffm = FeatureFusionModule(bot_ch + stage1_ch, bot_ch, stage1_ch)
        self.final = make_seg_head(in_ch=bot_ch,
                                   out_ch=num_classes,
                                   nums=head_nums,
                                   dropout=head_drop,
                                   bot_ch=head_ch)

        initialize_weights(self.final, self.ffm, self.bot_aspp)

    def forward(self, inputs):
        def norm(a):
            return torch.nn.functional.normalize(a, p=2, dim=1)

        feature, stage1_feature, gt = inputs
        x_size = gt.shape
        feature = self.bot_aspp(feature)
        stage1_feature = Upsample(stage1_feature, feature.shape[-2:])
        feature = norm(feature)

        b = feature.shape[0] // 2
        feat_U = feature[:b, :, :, :]
        feat_L = feature[b:, :, :, :]
        stage1_feat_L = norm(stage1_feature[b:, :, :, :])
        stage1_feat_U = norm(stage1_feature[:b, :, :, :])


        assert self.kl or self.kh
        if not self.kl:
            feat_L_t = feat_L
            feat_U_t = feat_U
        elif not self.kh:
            feat_L_t = torch.zeros_like(feat_L)
            feat_U_t = torch.zeros_like(feat_U)
            s1_ch = stage1_feat_L.shape[1]
            feat_L_t[:,:s1_ch,:] = stage1_feat_L
            feat_U_t[:,:s1_ch,:] = stage1_feat_U
        else:
            feat_L_t = norm(self.ffm(feat_L, stage1_feat_L))
            feat_U_t = norm(self.ffm(feat_U, stage1_feat_U))
        
        assert self.nl or self.nh
        if not self.nl:
            feat_L_hat = feat_U
        elif not self.nh:
            feat_L_hat = torch.zeros_like(feat_U)
            s1_ch = stage1_feat_L.shape[1]
            feat_L_hat[:,:s1_ch,:] = stage1_feat_L
        else:
            feat_L_hat = norm(self.ffm(feat_U, stage1_feat_L))

        pred_L_hat = self.final(feat_L_hat)
        pred_L_t = self.final(feat_L_t)
        B_mask = pred_L_t.argmax(1)
        pred_U_t = self.final(feat_U_t)
        I_mask = pred_U_t.argmax(1)

        inter_mask = (B_mask == I_mask).unsqueeze(1)

        pred_L_hat = Upsample(pred_L_hat, x_size[-2:])
        pred_L_t = Upsample(pred_L_t, x_size[-2:])
        # pred_U_t = Upsample(pred_U_t, x_size[2:])

        return pred_L_t, pred_L_hat, feat_L_t, feat_L_hat, feat_L_t * inter_mask, feat_U_t * inter_mask


class DCFMDecoder_flops(nn.Module):
    def __init__(self,
                 num_classes,
                 bot_ch=256,
                 head_drop=0.,
                 head_ch=256,
                 head_nums=1,
                 stage1_ch=128,
                 high_level_ch=720,
                 **kwargs):
        super(DCFMDecoder_flops, self).__init__()
        self.bot_ch = bot_ch

        if bot_ch > 0:
            self.bot_aspp = nn.Conv2d(high_level_ch,
                                    bot_ch,
                                    kernel_size=1,
                                    bias=False)
        else:
            self.bot_aspp = nn.Identity()
            bot_ch = high_level_ch
        self.ffm = FeatureFusionModule(bot_ch + stage1_ch, bot_ch, stage1_ch)
        self.final = make_seg_head(in_ch=bot_ch,
                                   out_ch=num_classes,
                                   nums=head_nums,
                                   dropout=head_drop,
                                   bot_ch=head_ch)

        initialize_weights(self.final, self.ffm, self.bot_aspp)

    def forward(self, inputs):
        def norm(a):
            return torch.nn.functional.normalize(a, p=2, dim=1)

        f1, f2, fs, x_size = inputs
        # f1 = self.bot_aspp(f1)
        # f2 = self.bot_aspp(f2)
        # fs = Upsample(fs, f1.shape[-2:])
        # fs = norm(fs)

        # f1 = norm(self.ffm(f1, fs))
        # f2 = norm(self.ffm(f2, fs))

        # f1 = Upsample(self.final(f1), x_size[-2:])
        f2 = Upsample(self.final(f2), x_size[-2:])
        # f = f1+f2
        return f2

if __name__ == "__main__":
    model = DCFMDecoder_flops(backbone="segformerb2", bot_ch=-1, head_drop=0.1, head_nums=0, high_level_ch=768,stage1_ch=32, num_classes=124).eval().cuda()
    f1 = torch.randn([1,256,120, 214]).cuda()
    f2 = torch.randn([1,256,120, 214]).cuda()
    fs = torch.randn([1,64,120, 214]).cuda()
    from thop import profile
    inputs = f1,f2,fs,(480,853)
    flops, params = profile(model, (inputs,))
    print(flops,params)
    f = model(inputs)
    print(f.shape)