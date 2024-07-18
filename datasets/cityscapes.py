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


class CSVdataset(Dataset):
    def __init__(self, path='/data/zyy/VSS/cityscapes_video', **kwargs):
        super(CSVdataset, self).__init__()
        self.split = kwargs.get('split')
        if self.split == "valid":
            self.split = "val"
        self.image_size = kwargs.get("size", 512)
        self.test_size = kwargs.get("test_size", 1024)
        self.wh = kwargs.get("wh", 1.)
        self.R = kwargs.get("R", 1)
        assert self.split is not None
        self.ann_path = os.path.join(path, "gtFine", self.split)
        self.img_path = os.path.join(path, "leftImg8bit", self.split)

        self.file_list = []
        print(self.ann_path)
        for root, _, filenames in os.walk(self.ann_path):
            for filename in filenames:
                folder, a, b, _, tp = filename.split('.')[0].split('_')
                if tp != "labelTrainIds":
                    continue
                self.file_list.append([folder, a, b])
        self.check_data()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

    def check_data(self):
        for (folder, a, b) in tqdm(self.file_list):
            org_path = os.path.join(
                self.img_path, folder,
                folder + "_" + a + "_" + b + "_leftImg8bit.png")
            gtf_path = os.path.join(
                self.ann_path, folder,
                folder + "_" + a + "_" + b + "_gtFine_labelTrainIds.png")
            assert os.path.exists(org_path) and os.path.exists(gtf_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, id):
        seed = np.random.randint(2147483647)
        set_seed(seed)
        scale = random.random() * 1.5 + 0.5
        h, w = int(self.image_size * scale), int(self.image_size * scale *
                                                 self.wh)
        if self.split == "train":
            transform = transforms.Compose([
                transforms.RandomRotation(10, fill=255),
                transforms.RandomCrop([h, w], pad_if_needed=True, fill=0),
                transforms.Resize(
                    [self.image_size,
                     int(self.image_size * self.wh)],
                    InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_mask = transforms.Compose([
                transforms.RandomRotation(10, fill=255),
                transforms.RandomCrop([h, w], pad_if_needed=True, fill=255),
                transforms.Resize(
                    [self.image_size,
                     int(self.image_size * self.wh)],
                    InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.test_size),
                transforms.ToTensor(),
            ])
            transform_mask = transforms.Compose([
                # transforms.Resize(self.test_size, InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        (folder, a, b) = self.file_list[id]

        org_path = os.path.join(
            self.img_path, folder,
            folder + "_" + a + "_" + b + "_leftImg8bit.png")
        img_org = Image.open(org_path)
        if self.split == "train":
            img_org = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)(img_org)
        set_seed(seed)
        img_org = self.trans_img(transform(img_org))

        r = random.randint(-self.R + 1, self.R)
        if r <= 0:
            r -= 1
        now_id = int(b)
        ref_id = now_id + r
        ref_id = str(ref_id).zfill(6)
        pre_path = os.path.join(
            self.img_path, folder,
            folder + "_" + a + "_" + ref_id + "_leftImg8bit.png")
        img_ref = Image.open(pre_path)
        set_seed(seed)
        img_ref = self.trans_img(transform(img_ref))

        gtf_path = os.path.join(
            self.ann_path, folder,
            folder + "_" + a + "_" + b + "_gtFine_labelTrainIds.png")
        try:
            img_gtf = Image.open(gtf_path)
            set_seed(seed)
            img_gtf = transform_mask(img_gtf)
        except:
            print(gtf_path)
        img_gtf = (img_gtf * 255).round().long().squeeze(0)
        return img_org, img_ref, img_gtf, org_path.split('/')[-1]


class test_CSVdataset():
    def __init__(self, **kwargs) -> None:
        path = kwargs.get('path')
        self.use_flip = kwargs.get('use_flip')
        self.multi_scale = kwargs.get("multi_scale")
        self.image_size = kwargs.get("size")
        self.R = kwargs.get("R", -1)

        self.ann_path = os.path.join(path, "gtFine", "val")
        self.img_path = os.path.join(path, "leftImg8bit", "val")

        self.file_list = []
        for root, _, filenames in os.walk(self.ann_path):
            for filename in filenames:
                folder, a, b, _, tp = filename.split('.')[0].split('_')
                if tp != "labelTrainIds":
                    continue
                self.file_list.append([folder, a, b])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.file_list)

    def get_img(self, img, scale):
        ret = transforms.Resize(round(self.image_size * scale))(img)
        ret = transforms.ToTensor()(ret)
        ret = self.trans_img(ret)
        if self.use_flip:
            ret_flip = functional.hflip(ret)
            ret = torch.cat([ret.unsqueeze(0), ret_flip.unsqueeze(0)], 0)
        else:
            ret = ret.unsqueeze(0)
        return ret

    def get_video(self, id):
        (folder, a, b) = self.file_list[id]
        gtf_path = os.path.join(
            self.ann_path, folder,
            folder + "_" + a + "_" + b + "_gtFine_labelTrainIds.png")
        img_gtf = Image.open(gtf_path)
        img_gtf = transforms.ToTensor()(img_gtf)
        img_gtf = (img_gtf * 255).round().long()

        frames = []
        sims = []
        frame_path = []

        if self.multi_scale:
            scales = [1.0, 0.5, 0.75, 1.25, 1.5, 1.75]
        else:
            scales = [1.0]
        if self.R == -1:
            s = int(b) - 19
            t = s + 30
        else:
            s = int(b) - self.R
            t = int(b) + self.R + 1
        for i in range(s, t):
            img_path = os.path.join(
                self.img_path, folder,
                folder + "_" + a + "_" + str(i).zfill(6) + "_leftImg8bit.png")
            frame_path.append(img_path)
            img = Image.open(img_path)

            imgs = []
            for scale in scales:
                imgs.append(self.get_img(img, scale))
            if len(frames) > 0:
                pre_img = frames[-1]
                sims.append((imgs[0] - pre_img[0]).abs().mean())
            frames.append(imgs)

        gt = {int(b) - s: img_gtf}
        return scales, frames, gt, (1024, 2048), sims, frame_path


def get_dataset(test_video=False, **kwargs):
    if test_video:
        return test_CSVdataset(**kwargs)
    return CSVdataset(**kwargs)


if __name__ == "__main__":
    d = CSVdataset(split="train", use_grad=False)
    # print(len(stat.folder), label_pool.count)
    print(len(d))
    print(d[0][0][0].shape)
