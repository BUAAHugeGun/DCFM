import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import random
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


class CamViddataset(Dataset):

    def __init__(self, path='/home/LAB/r-yangwangwang/data/camvid', **kwargs):
        super(CamViddataset, self).__init__()
        self.path = path
        self.mode = kwargs.get('mode', "video")
        self.split = kwargs.get('split')
        if self.split == "valid":
            if kwargs.get("test", False):
                self.split = "test"
            else:
                self.split = "val"
        self.image_size = kwargs.get("size", 480)
        self.R = kwargs.get("R", 1)
        assert self.split is not None
        self.images = []
        self.id2id = {}
        tot = 0
        with open(os.path.join(path, "id2id.txt")) as f:
            for line in f.readlines():
                line = line.replace('\n', '').split(' ')
                self.id2id[line[0]] = line[1]

        with open(os.path.join(path, self.split + ".txt"), "r") as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                filename = self.id2id[line]
                self.images.append([line, line.split('_')[0], filename])
        self.check_data()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

        random.seed(2333)
        # print(self.id2vd)

    def check_data(self):
        for (org_name, folder, filename) in tqdm(self.images):
            org_path = os.path.join(self.path, folder, filename + ".png")
            gtf_path = os.path.join(self.path, "gt", org_name + "_L.png")
            if not os.path.exists(org_path):
                print(org_path)
            if not os.path.exists(gtf_path):
                print(gtf_path)
            assert os.path.exists(org_path) and os.path.exists(gtf_path)

    def __len__(self):
        return len(self.images)

    def get(self, org_name, folder, filename, mode):
        if mode == "org":
            org_path = os.path.join(self.path, folder, filename + ".png")
            img_org = Image.open(org_path)
            set_seed(self.seed)
            if self.split == "train":
                img_org = self.transform(img_org)
                img_org = transforms.Resize(
                    [self.image_size, self.image_size],
                    InterpolationMode.BILINEAR)(img_org)
            img_org = self.trans_img(transforms.ToTensor()(img_org))
            return img_org
        elif mode == "gt":
            gtf_path = os.path.join(self.path, "gt", org_name + "_L.png")
            img_gtf = Image.open(gtf_path).convert("L")
            set_seed(self.seed)
            if self.split == "train":
                img_gtf = self.transform(img_gtf)
                img_gtf = transforms.Resize([self.image_size, self.image_size],
                                            InterpolationMode.NEAREST)(img_gtf)
            img_gtf = transforms.ToTensor()(img_gtf)

            img_gtf = (img_gtf * 255).round().long()
            img_gtf[img_gtf >= 11] = 255
            return img_gtf.squeeze(0)

    def __getitem__(self, id):
        self.seed = np.random.randint(2147483647)
        if self.split == "train":
            self.transform = transforms.Compose([
                transforms.RandomRotation(10, fill=255),
                transforms.RandomCrop(self.image_size *
                                      (random.random() * 0.5 + 0.5)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([])

        (org_name, folder, filename) = self.images[id]
        gtf = self.get(org_name, folder, filename, "gt")
        org = self.get(org_name, folder, filename, "org")
        r = random.randint(1 - self.R, self.R)
        if r <= 0:
            r -= 1
        num = int(filename.split('_')[-1])
        num_ref = max(num + r, 1)
        num_ref = str(num + r).zfill(6)
        filename_ref = '_'.join([filename.split('_')[0], num_ref])
        ref_org = self.get(org_name, folder, filename_ref, "org")

        if self.mode == "video":
            return org, ref_org, gtf, filename
            # return [torch.cat([ref_org, org], 0), gtf, filename]
        else:
            return [org, gtf, filename]


class test_CamViddataset():
    def __init__(self, path, **kwargs) -> None:
        self.path = path
        self.mode = kwargs.get('mode', "video")
        self.multi_scale = kwargs.get("multi_scale")
        self.image_size = kwargs.get("size", 480)
        self.R = kwargs.get("R", 1)
        self.images = []
        self.id2id = {}
        tot = 0
        with open(os.path.join(path, "id2id.txt")) as f:
            for line in f.readlines():
                line = line.replace('\n', '').split(' ')
                self.id2id[line[0]] = line[1]

        with open(os.path.join(path, "test" + ".txt"), "r") as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                filename = self.id2id[line]
                self.images.append([line, line.split('_')[0], filename])
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

        random.seed(2333)

    def __len__(self):
        return len(self.images)

    def get(self, org_name, folder, filename, mode):
        if mode == "org":
            org_path = os.path.join(self.path, folder, filename + ".png")
            img_org = Image.open(org_path)
            img_org = self.trans_img(transforms.ToTensor()(img_org))
            return img_org
        elif mode == "gt":
            gtf_path = os.path.join(self.path, "gt", org_name + "_L.png")
            img_gtf = Image.open(gtf_path).convert("L")
            img_gtf = transforms.ToTensor()(img_gtf)

            img_gtf = (img_gtf * 255).round().long()
            img_gtf[img_gtf >= 11] = 255
            return img_gtf

    def get_video(self, id):
        (org_name, folder, filename) = self.images[id]
        gtf = self.get(org_name, folder, filename, "gt")
        num = int(filename.split('_')[-1])

        frames = []
        sims = []
        frame_path = []
        if self.multi_scale:
            scales = [1.0, 0.5, 0.75, 1.25, 1.5, 1.75]
        else:
            scales = [1.0]
        if self.R == -1:
            s = int(num) - 19
            t = s + 30
        else:
            s = int(num) - min(self.R, 1)
            t = int(num) + self.R + 1

        for i in range(s, t):
            filename_frame = '_'.join(
                [filename.split('_')[0], str(i).zfill(6)])
            img = self.get(org_name, folder, filename_frame, "org")

            imgs = []
            imgs.append(img.unsqueeze(0))
            if len(frames) > 0:
                pre_img = frames[-1]
                sims.append((imgs[0] - pre_img[0]).abs().mean())
            frames.append(imgs)

        gt = {int(num) - s: gtf}
        return scales, frames, gt, (720, 960), sims, frame_path


def get_dataset(test_video=False, **kwargs):
    if test_video:
        return test_CamViddataset(**kwargs)
    return CamViddataset(**kwargs)


if __name__ == "__main__":
    dataset = CamViddataset(mode="image", split="train", size=640)
    org, gtf, filename = dataset[1]
    org = transforms.ToPILImage()(org)
    gtf = transforms.ToPILImage()(gtf / 255)
    # org.save("org.png")
    # gtf.save("gtf.png")
