# DCFM
Deep Common Feature Mining for Efficient Video Semantic Segmentation

This repository is the official implementation of "Deep Common Feature Mining for Efficient Video Semantic Segmentation".  This paper is under submission, we will show it later.

## Install & Requirements

Please follow the guidelines in [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

Requirements:
```
python == 3.8
pytorch >= 2.0.0
mmcv == 1.7.1
mmsegmentation == 0.30.0
tensorboardX
tqdm
yacs
ftfy
regex
timm
scikit-learn
```
## Usage

### Data preparation

Please follow [VSPW](https://github.com/sssdddwww2/vspw_dataset_download) to download VSPW 480P dataset.

Please follow [Cityscapes](https://www.cityscapes-dataset.com/) to download Cityscapes dataset. 

Please follow [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) to download CamVid dataset.

After correctly downloading, the file system is as follows:
```
<dataset path>
├── VSPW_480p
│   ├── train.txt
|   ├── val.txt
|   ├── test.txt
|   └── data
│       ├── video1
│       ├── video2
│       ├── ...
│
├── cityscapes_video
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
├── camvid
│   ├── 0001TP
│   ├── 0006R0
│   ├── 0016E5
│   ├── Seq05VD
│   ├── gt
│   ├── test.txt
│   ├── train.txt
│   ├── val.txt
```
The dataset should be put in `DCFM/data`. Or you can use Symlink:
```
cd DCFM
ln -s <dataset path> ./data
```

### Pretrained models

Download weights ([google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)|[onedrive](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Fpretrained%5Fmodels&ga=1)) pretrained on ImageNet-1K (provided by SegFormer), and put them in a folder pretrained/.

Download weights ([resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth), [resnet50&101](https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v)) of ResNet.

Place the `.pth` files in `DCFM/pretrain` directory.

## Acknowledgement
The code is heavily based on the following repositories:

- https://github.com/GuoleiSun/VSS-CFFM

- https://github.com/NVlabs/SegFormer

- https://github.com/hszhao/semseg/

Thanks for their amazing works.

### Training
```shell
torchrun --nproc_per_node=4 --master_port=<port> ./tools/train.py --root ./exp/vspw/dcfm/mitb2 --gpus 0,1,2,3
```
The checkpoints and output information generated during training will be saved in the `<exp_folder>/logs` directory under the experiment directory, such as `./exp/vspw/dcfm/mitb2/logs`.
### Test
Download the trained weights from[here](), and place the `*.pth` files in `<exp_folder>/logs/`.
```shell
python ./tools/test.py --root ./exp/vspw/dcfm/mitb2/infer.yaml --k 2
```
The output generated during the testing process will be saved in a text file with the same name under the experiment directory, and the final testing results will be incrementally saved in `results.txt` under the experiment directory, such as `./exp/vspw/dcfm/mitb2/infer.txt` and `./exp/vspw/dcfm/mitb2/results.txt`.