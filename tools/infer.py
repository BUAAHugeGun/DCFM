import sys
sys.path.append("./")
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import optim
from process.build import build_processes
from tools.utils import init_seeds, open_config, get_parameter_number, load_ckpt
from tools.to_log import to_log, set_file
import yaml
import numpy as np
from torch.backends import cudnn
import random
from torch import distributed as dist
import torch
import os
import argparse



def det_ckpts(path, thr):
    ret = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[-1] == "pth":
                it = filename.split('.')[-2].split('_')[-1]
                it = int(it)
                if it not in ret and it > thr:
                    ret.append(it)
    return sorted(ret)


def infer(args, root, name, k=-1):
    if k != -1:
        args["processes"]["ED"]["K"] = k
    output_file = open(os.path.join(root, name + ".txt"), "w")
    set_file(output_file, 0)
    args_test = args['test']

    if args_test['load_iter'] == -1:
        load_iters = det_ckpts(root, args_test['iter_thr'])
    else:
        load_iters = [args_test['load_iter']]

    print("test iters: ", load_iters)
    for load_iter in load_iters:
        
        args_processes = args['processes']
        data_pool = {}
        processes = build_processes(
            data_pool=data_pool, local_rank=0, **args_processes)
        nums_videos = data_pool['nums_videos']

        for process in processes:
            process.load(root=root, iter=load_iter)
        for video_i in range(nums_videos):
            torch.cuda.empty_cache()
            data_pool.clear()
            data_pool['video_id'] = video_i
            to_log(str(video_i) + ' / ' + str(nums_videos))
            for c in range(args_test.get('cicle', 1)):
                for process in processes:
                    process.run()

                for name, loss in data_pool['metric'].items():
                    to_log("{}: {:.5f}".format(name, loss))

        with open(os.path.join(root, "results.txt"), "a") as f:
            print("k=", args["processes"]["ED"]["K"], file=f)
            print("load iter: {}".format(load_iter), file=f)
            for name, loss in data_pool['metric'].items():
                print("{}: {:.5f}".format(name, loss), file=f)
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--k", type=int, default=-1)
    args = parser.parse_args()

    f = open(args.root)
    name = args.root.split('/')[-1].split('.')[0]
    folder_path = '/'.join(args.root.split('/')[:-1])
    infer(yaml.load(f, Loader=yaml.FullLoader), folder_path, name, args.k)
