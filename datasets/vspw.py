import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms, functional
from PIL import Image
import torch
import random
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


class VSPWdataset(Dataset):
    def __init__(self, path, **kwargs):
        super(VSPWdataset, self).__init__()
        self.path = path
        self.mode = kwargs.get('mode', "all")
        self.val_mode = kwargs.get('val_mode', "random")
        self.self_thr = kwargs.get('self_thr', 0.)
        self.split = kwargs.get('split')
        self.R = kwargs.get("R", 1)
        if self.split == "valid":
            self.split = 'val'
            self.R = 1
        self.image_size = kwargs.get("size", 480)
        assert self.split is not None
        self.videos = []
        self.id2vd = []
        self.val_id = []
        self.inval_id = []
        tot = 0
        with open(os.path.join(path, self.split + ".txt"), "r") as f:
            folder_list = f.readlines()
            for folder in folder_list:
                folder = folder[:-1]
                video = [folder, tot, []]
                # mask png/org jpg
                for root, _, filenames in os.walk(
                        os.path.join(path, "data", folder, "origin")):
                    for filename in filenames:
                        name = filename.split('.')[0]
                        if name == "":
                            continue
                        video[2].append(int(name))
                        self.id2vd.append(len(self.videos))
                        tot += 1
                video[2] = sorted(video[2])
                self.videos.append(video)
        # self.check_data()
        for id in range(len(self.id2vd)):
            vid = self.id2vd[id]
            (folder, tot, filenames) = self.videos[vid]
            length = len(filenames)
            image_id = id - tot
            if self.split == "train" and image_id < int(length * self.self_thr):
                self.inval_id.append(id)
            else:
                self.val_id.append(id)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

    def check_data(self):
        for (folder, tot, filenames) in tqdm(self.videos):
            for filename in filenames:
                org_path = os.path.join(self.path, "data", folder, "origin",
                                        filename + ".jpg")
                gtf_path = os.path.join(self.path, "data", folder, "mask",
                                        filename + ".png")
                if not os.path.exists(org_path):
                    print(org_path)
                if not os.path.exists(gtf_path):
                    print(gtf_path)
                assert os.path.exists(org_path) and os.path.exists(gtf_path)

    def __len__(self):
        if self.mode == 'half' and self.split == "train":
            return int(1.5 * len(self.val_id))
        if self.val_mode == 'all' and self.split == "val":
            return len(self.val_id)
        if self.mode == 'all' and self.split == "train":
            return len(self.val_id)
        return len(self.videos)

    def get(self, folder, filename, no_gt=False):
        filename = str(filename).zfill(8)
        org_path = os.path.join(self.path, "data", folder, "origin",
                                filename + ".jpg")

        img_org = Image.open(org_path).convert("RGB")
        if self.split == "train":
            img_org = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(img_org)
        set_seed(self.seed)
        img_org = self.transform(img_org)
        img_org = self.trans_img(transforms.ToTensor()(img_org))

        gtf_path = os.path.join(self.path, "data", folder, "mask",
                                filename + ".png")
        img_gtf = Image.open(gtf_path)
        set_seed(self.seed)
        img_gtf = self.transform_gtf(img_gtf)
        img_gtf = transforms.ToTensor()(img_gtf)

        img_gtf = (img_gtf * 255).round().long()
        img_gtf[img_gtf > 124] = 256
        img_gtf[img_gtf == 0] = 256
        img_gtf = img_gtf - 1
        img_gtf = img_gtf.squeeze(0)
        if self.split == "train" and no_gt:
            img_gtf[:] = 255
        return img_org, img_gtf, org_path.split('/')[-1]

    def __getitem__(self, id):
        self.seed = np.random.randint(2147483647)
        set_seed(self.seed)
        if self.split == "train":
            ratio = random.random() * 1.5 + 0.5
            self.transform = transforms.Compose([
                transforms.Resize(
                    [int(480 * ratio), int(853 * ratio)],
                    InterpolationMode.BILINEAR),
                transforms.RandomRotation([-20, 20]),
                transforms.RandomCrop((480, 853), pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
            ])
            self.transform_gtf = transforms.Compose([
                transforms.Resize(
                    [int(480 * ratio), int(853 * ratio)],
                    InterpolationMode.NEAREST),
                transforms.RandomRotation([-20, 20], fill=255),
                transforms.RandomCrop(
                    (480, 853), pad_if_needed=True, fill=255),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize([480, 853], InterpolationMode.BILINEAR)])
            self.transform_gtf = transforms.Compose(
                [transforms.Resize([480, 853], InterpolationMode.NEAREST)])

        if self.__len__() == len(self.videos):
            (folder, tot, filenames) = self.videos[id]
            length = len(filenames)
            if self.split == "val":
                image_id = length // 2
            else:
                image_id = random.randint(self.R, length - self.R - 1)
        else:
            if id >= len(self.val_id):
                id = random.randint(0, len(self.inval_id) - 1)
                id = self.inval_id[id]
            else:
                id = self.val_id[id]
            vid = self.id2vd[id]
            (folder, tot, filenames) = self.videos[vid]
            length = len(filenames)
            image_id = id - tot
        ref_id = random.randint(1 - self.R, self.R)
        if ref_id <= 0:
            ref_id -= 1
        ref_id += image_id
        ref_id = min(max(ref_id, 0), length - 1)

        video_thr = int(self.self_thr * length)
        ref_org, ref_gtf, ref_filename = self.get(
            folder, filenames[ref_id], ref_id < video_thr)
        org, gtf, filename = self.get(
            folder, filenames[image_id], image_id < video_thr)
        # print(folder, filename, ref_filename)
        return org, ref_org, gtf, filename


class test_VSPWdataset():
    def __init__(self, **kwargs) -> None:
        self.path = kwargs.get('path')
        self.use_flip = kwargs.get('use_flip')
        self.multi_scale = kwargs.get("multi_scale")
        self.image_size = 480
        self.random = kwargs.get("random", True)
        self.R = kwargs.get("R", -1)

        self.videos = []
        tot = 0
        with open(os.path.join(self.path, "val" + ".txt"), "r") as f:
            folder_list = f.readlines()
            for folder in folder_list:
                folder = folder[:-1]
                video = [folder, tot, []]
                # mask png/org jpg
                for root, _, filenames in os.walk(
                        os.path.join(self.path, "data", folder, "origin")):
                    for filename in filenames:
                        name = filename.split('.')[0]
                        if name == "":
                            continue
                        video[2].append(int(name))
                        tot += 1
                video[2] = sorted(video[2])
                self.videos.append(video)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.videos)

    def get_gt(self, folder, filename):
        filename = str(filename).zfill(8)
        gtf_path = os.path.join(self.path, "data", folder, "mask",
                                filename + ".png")
        img_gtf = Image.open(gtf_path)
        img_gtf = transforms.Resize(
            [480, 853], InterpolationMode.NEAREST)(img_gtf)
        img_gtf = transforms.ToTensor()(img_gtf)

        img_gtf = (img_gtf * 255).round().long()
        img_gtf[img_gtf > 124] = 256
        img_gtf[img_gtf == 0] = 256
        img_gtf = img_gtf - 1
        return img_gtf

    def get_img(self, folder, filename, scale):
        filename = str(filename).zfill(8)
        org_path = os.path.join(self.path, "data", folder, "origin",
                                filename + ".jpg")
        img_org = Image.open(org_path).convert("RGB")
        img_org = transforms.Resize(
            [round(480 * scale), round(853 * scale)], InterpolationMode.BILINEAR)(img_org)
        img_org = self.trans_img(transforms.ToTensor()(img_org))
        if self.use_flip:
            img_flip = functional.hflip(img_org)
            img = torch.cat([img_org.unsqueeze(0), img_flip.unsqueeze(0)], 0)
        else:
            img = img_org.unsqueeze(0)
        return img

    def get_video(self, video_id):
        (folder, tot, filenames) = self.videos[video_id]
        length = len(filenames)
        if self.random:
            if self.R == -1:
                gt_slt = [random.randint(0, length - 1)]
                img_slt = [i for i in range(length)]
            else:
                gt_slt = [random.randint(self.R, length - 1 - self.R)]
                img_slt = [i for i in range(
                    gt_slt[0] - self.R, gt_slt[0] + self.R + 1)]
        else:
            gt_slt = [i for i in range(length)]
            img_slt = [i for i in range(length)]

        frames = []
        sims = []
        gt = {}
        if self.multi_scale:
            scales = [1.0, 0.5, 0.75, 1.25, 1.5, 1.75]
        else:
            scales = [1.0]

        for index, gt_id in enumerate(gt_slt):
            gt[gt_id - img_slt[0]] = self.get_gt(folder, filenames[gt_id])
        for index, img_id in enumerate(img_slt):
            imgs = []
            for scale in scales:
                img = self.get_img(folder, filenames[img_id], scale)
                imgs.append(img)
            if len(frames) > 0:
                pre_img = frames[-1]
                sims.append((imgs[0] - pre_img[0]).abs().mean())
            frames.append(imgs)

        return scales, frames, gt, (480, 853), sims


def get_dataset(test_video=False, **kwargs):
    if test_video:
        return test_VSPWdataset(**kwargs)
    return VSPWdataset(**kwargs)


if __name__ == "__main__":
    d = VSPWdataset(split="train")
    # print(len(stat.folder), label_pool.count)
    print(len(d))
    print(d[0][0].shape, d[0][1].shape)
