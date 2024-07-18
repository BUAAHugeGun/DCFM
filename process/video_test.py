from torch.utils.data import DataLoader
from torch.utils.data import distributed
from datasets import cityscapes, vspw, camvid
from nets import dcfm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import torch
from tools.utils import load_ckpt, get_parameter_number
import tools.utils
from tools.to_log import to_log
import os
from torch import optim
from metric import get_metric
from torch import nn
import time
from tqdm import tqdm


class video_test_base_process():
    def __init__(
        self,
        data_pool,
    ) -> None:
        self.data_pool = data_pool
        self.input_name = []
        self.output_name = []
        if torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')

    def run(self, **kwargs):
        self.put_data(self.infer())

    def infer(self, **kwargs):
        pass

    def load(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass

    def get_data(self):
        ret = []
        for name in self.input_name:
            ret.append(self.data_pool[name])
        return ret

    def put_data(self, *_data):
        while len(self.output_name) != len(_data):
            _data = _data[0]
        for id, name in enumerate(self.output_name):
            names = name.split('/')
            now = self.data_pool
            for d in names[:-1]:
                if d not in now:
                    now[d] = {}
                now = now[d]
            try:
                _d = _data[id].to(self.device)
            except:
                _d = _data[id]
            now[names[-1]] = _d


class video_test_data_process(video_test_base_process):
    def __init__(self, data_pool, **kwargs) -> None:
        super(video_test_data_process, self).__init__(data_pool)
        self.input_name = kwargs.get('input_name', [])
        self.output_name = kwargs['output_name']
        self.dataset = getattr(sys.modules[__name__], kwargs['tag'])
        self.dataset = self.dataset.get_dataset(test_video=True,
                                                **kwargs['args'])
        self.data_pool['nums_videos'] = len(self.dataset)
        self.video_id = -1

    def infer(self):
        id = self.get_data()[0]
        return self.dataset.get_video(id)


class video_test_model_process(video_test_base_process):
    def __init__(self, data_pool, **kwargs) -> None:
        super(video_test_model_process, self).__init__(data_pool)
        self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']

        self.ckpt_dict = {}
        self.encoder = self.get_model(**kwargs['encoder'])
        self.decoder = self.get_model(**kwargs['decoder'])

        self.backbone_name = kwargs['encoder']['args']['backbone']
        self.stage1 = kwargs['encoder']['args'].get('stage1', )
        self.mode = kwargs['mode']
        self.num_classes = kwargs['decoder']["args"]['num_classes']

        self.aks_max = kwargs.get("aks_max", 5)
        self.aks_thr = kwargs.get("aks_thr", 0)
        self.aks = kwargs.get("aks", False)
        self.GOP_size = kwargs.get('K', 1)
        self.val_I = kwargs.get("val_I", True)
        self.val_B = kwargs.get("val_B", True)
        self.val_pos = kwargs.get("val_pos", True)
        self.st = 0
        self.infer_time = []
        print(get_parameter_number(self.encoder))
        print(get_parameter_number(self.decoder))

        with torch.no_grad():
            x = torch.randn((1, 3, 1024, 2048)).to(self.device)
            _ = self.encoder.backbone(x)
    
    def get_model(self, **kwargs):
        model_name = kwargs['tag']
        ckpt_name = kwargs['ckpt']
        model = getattr(sys.modules[__name__], model_name)
        model = model.get_model(**kwargs.get("args", {}))
        model.require_grad = False
        model = model.to(self.device).eval()
        self.ckpt_dict[ckpt_name] = model
        return model


    def load(self, root, iter):
        load_iter = load_ckpt(self.ckpt_dict,
                              iter,
                              root,
                              rm_ddp=True)
        assert load_iter == iter

    def cal_flip(self, pred):
        if pred.shape[0] == 1:
            return pred
        elif pred.shape[0] == 2:
            p1 = pred[0:1, :, :, :]
            p2 = pred[1:2, :, :, :].flip(-1)
            pred = (p1 + p2)
            return pred

    @staticmethod
    def norm(a):
        if a is None:
            return None
        return torch.nn.functional.normalize(a, p=2, dim=1)

    @staticmethod
    def Upsample(x, size):
        return nn.functional.interpolate(x,
                                         size=size,
                                         mode='bilinear',
                                         align_corners=False)

    def image_forward(self, x, image_size):
        x = self.encoder.backbone(x)
        x = self.decoder.final(self.norm(x))
        x = self.Upsample(x, image_size)
        return x

    def align_shape(self, a, b):
        if a.shape[-2:] != b.shape[-2:]:
            return self.Upsample(a, b.shape[-2:])
        return a

    def merge(self, pred_pre, pred_post):
        if pred_pre is None:
            return pred_post
        if pred_post is None:
            return pred_pre
        return (pred_pre + pred_post)  # / 2

    def ffm(self, fI, fB):
        if fI is None:
            return None
        fB = self.align_shape(fB, fI)
        return self.decoder.ffm(fI, fB)

    def seg_head(self, feat, x_size):
        if feat is None:
            return None
        pred = self.decoder.final(feat)
        pred = self.Upsample(pred, x_size)
        return pred

    def I_forward(self, frame, image_size):
        feature = self.encoder.backbone(frame)
        aspp = feature
        aspp = self.decoder.bot_aspp(aspp)
        aspp = self.norm(aspp)  #

        stage1_feat = self.norm(self.encoder.stage1_feature)
        stage1_feat = self.align_shape(stage1_feat, aspp)
        if not self.decoder.kl:
            feat = aspp
        elif not self.decoder.kh:
            feat = torch.zeros_like(aspp)
            s1_ch = stage1_feat.shape[1]
            feat[:,:s1_ch,:] = stage1_feat
        else:
            feat = self.norm(self.decoder.ffm(aspp, stage1_feat))
        pred = self.seg_head(feat, image_size)
        return pred, aspp

    def B_forward(self, frame):
        if self.backbone_name[:7] == "hrnetv2":
            x = self.encoder.backbone.conv1(frame)
            x = self.encoder.backbone.bn1(x)
            x = self.encoder.backbone.relu(x)
            x = self.encoder.backbone.conv2(x)
            x = self.encoder.backbone.bn2(x)
            x = self.encoder.backbone.relu(x)
            x = self.encoder.backbone.layer1(x)
        elif self.backbone_name[:6] == "pspnet":
            x = self.encoder.backbone.layer0(frame)
            x = self.encoder.backbone.layer1(x)
        elif self.backbone_name[:6] == "resnet":
            x = self.encoder.backbone.layer0(frame)
            x = self.encoder.backbone.layer1(x)
        elif self.backbone_name[:9] == "segformer":
            B = frame.shape[0]
            x, H, W = self.encoder.backbone.encoder.patch_embed1(frame)
            if self.stage1 != "vit0":
                for i, blk in enumerate(self.encoder.backbone.encoder.block1):
                    x = blk(x, H, W)
                    if self.stage1 == "vit1":
                        break
                if self.stage1 == "vit3":
                    x = self.encoder.backbone.encoder.norm1(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.norm(x)

    def sim(self, fpos, fpre):
        score = (fpre-fpos).abs().mean()
        return score

    @torch.no_grad()
    def infer(self):
        frames, image_size, sims = self.get_data()
        len_video = len(frames)

        p_size = [1, self.num_classes, image_size[0], image_size[1]]
        preds = [None for _ in range(len_video)]
        self.infer_time = []
        if self.mode == "image":
            for frame_i in tqdm(range(len_video)):
                pred = torch.zeros(p_size)
                fs = frames[frame_i]

                # for scale_i in range(len(fs)):
                #     fs[scale_i] = fs[scale_i].cuda()

                f_time = 0.

                for scale_i in range(len(fs)):
                    frame = fs[scale_i].cuda()
                    torch.cuda.synchronize()
                    t1 = time.time()
                    x = self.image_forward(frame, image_size)
                    torch.cuda.synchronize()
                    t2 = time.time()
                    f_time += t2 - t1

                    pred += self.cal_flip(x.cpu())

                self.infer_time.append(f_time)

                preds[frame_i] = pred.cpu()

                # for scale_i in range(len(fs)):
                #     fs[scale_i] = fs[scale_i].cpu()
        else:
            I_feats = {}
            I_feats[None] = None
            pre_I = None
            pos_I = self.st

            for id in tqdm(range(len_video)):
                if pos_I not in I_feats.keys():
                    pred = torch.zeros(p_size)
                    frame = frames[pos_I][0].cuda()

                    torch.cuda.synchronize()
                    t1 = time.time()
                    pp, I_feat = self.I_forward(frame, image_size)
                    torch.cuda.synchronize()
                    t2 = time.time()
                    self.infer_time.append(t2 - t1)

                    pred += self.cal_flip(pp.cpu())
                    if self.val_I:
                        preds[pos_I] = pred
                    I_feats[pos_I] = I_feat

                if id == pos_I:  # I frames
                    if pre_I is not None:
                        I_feats.pop(pre_I)
                    pre_I = pos_I
                    pos_I += self.GOP_size
                    if pos_I >= len_video:
                        pos_I = None
                    elif self.aks:
                        while self.sim(frames[pos_I][0], frames[pre_I][0]) < self.aks_thr and pos_I < len_video - 1 and pos_I - pre_I < self.aks_max:
                            pos_I += 1
                        print(self.sim(frames[pos_I][0], frames[pre_I][0]), pos_I, pre_I)
                    # else:
                    #     print(self.sim(frames[pos_I][0], frames[pre_I][0]))

                else:  # B frames
                    pred = torch.zeros(p_size)
                    frame = frames[id][0].cuda()

                    torch.cuda.synchronize()
                    t1 = time.time()
                    B_feat = self.B_forward(frame)

                    if not self.decoder.nl:
                        feat_pre = I_feats[pre_I]
                        feat_pos = I_feats[pos_I]
                    elif not self.decoder.nh:
                        s1_ch = B_feat.shape[1]
                        if I_feats[pre_I] is not None:
                            feat_pre = torch.zeros_like(I_feats[pre_I])
                        else:
                            feat_pre = torch.zeros_like(I_feats[pos_I])
                        feat_pre[:,:s1_ch,:] = B_feat
                        feat_pos = feat_pre.clone()
                    else:
                        feat_pre = self.norm(self.ffm(I_feats[pre_I], B_feat))
                        if self.val_pos:
                            feat_pos = self.norm(self.ffm(I_feats[pos_I], B_feat))
                        else:
                            feat_pos = None

                    pred_pre = self.seg_head(feat_pre,
                        image_size)
                    pred_pos = self.seg_head(feat_pos,
                        image_size)
                    pp = self.merge(pred_pre, pred_pos)
                    torch.cuda.synchronize()
                    t2 = time.time()
                    self.infer_time.append(t2 - t1)
                    pred += self.cal_flip(pp.cpu())
                    if self.val_B:
                        preds[id] = pred
                        
            if self.val_pos:
                self.st = (self.st + 1) % self.GOP_size


        torch.cuda.empty_cache()
        return preds, self.infer_time

class video_test_save_process(video_test_base_process):
    def __init__(self, data_pool, **kwargs) -> None:
        super(video_test_save_process, self).__init__(data_pool)
        self.input_name = kwargs['input_name']
        self.output_name = []
        self.video_test = kwargs.get('video_test', False)
        self.save_path = kwargs.get("save_path", "results")
        self.num_classes = kwargs.get("num_classes", 19)

        from cityscapesscripts.helpers.labels import labels
        from matplotlib import pyplot as plt
        def generate_colors(num_colors):
            colors = plt.cm.get_cmap('hsv', num_colors)
            rgb_colors = {}
            for i in range(num_colors):
                rgb = colors(i)[:3]
                rgb_colors[i] = tuple(int(255 * c) for c in rgb)
            return rgb_colors
        
        if self.num_classes != 19:
            self.id2color = generate_colors(self.num_classes)
        else:
            self.id2color = {}
            for label in labels:
                self.id2color[label.trainId] = label.color

    def infer(self):
        frames,preds,frame_path= self.get_data()
        import numpy as np
        import shutil
        from PIL import Image
        for i in range(len(frame_path)):
            img_path = frame_path[i]
            img_name = img_path.split('/')[-1].split('.')[0]
            shutil.copy(img_path, self.save_path)
            pred = preds[i]
            pred = pred.argmax(1).squeeze(0)
            p = np.zeros([pred.shape[-2], pred.shape[-1], 3])
            for k in self.id2color.keys():
                p[pred==k] = np.array(self.id2color[k])
            p = Image.fromarray(np.uint8(p))
            p.save(os.path.join(self.save_path, img_name+"_res.png"))
        return []
        # exit(0)

class video_test_metric_process(video_test_base_process):
    def __init__(self, data_pool, **kwargs) -> None:
        super(video_test_metric_process, self).__init__(data_pool)
        self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        self.model_name = kwargs['tag']
        self.video_test = kwargs.get('video_test', False)
        self.model = get_metric(self.model_name,
                                video_test=self.video_test,
                                **kwargs.get("args", {}))
        self.model.reset()

    def infer(self):
        x = self.get_data()
        self.model(*x)
        y = self.model.get()
        return y
