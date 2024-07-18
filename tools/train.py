import sys
sys.path.append("./")
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import optim
from process.build import build_processes
from tools.utils import init_seeds, open_config, get_parameter_number, load_ckpt
from tools.to_log import to_log, set_file
from torch.cuda.amp import autocast
import yaml
import numpy as np
from torch.backends import cudnn
import random
from torch import distributed as dist
import torch
import os
import argparse

world_size = 1
local_rank = 0
device = None
log_file = None


def train(args, root):
    args_train = args['train']
    seed = args_train.get('seed', 2106346)
    init_seeds(seed + local_rank, True)
    global log_file
    if local_rank == 0:
        if not os.path.exists(os.path.join(root, "logs/result/event")):
            os.makedirs(os.path.join(root, "logs/result/event"))
    dist.barrier()
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    set_file(log_file, rank=local_rank)
    to_log(args)

    args_processes = args['processes']
    data_pool = {"opt": {}, "sch": {}}
    processes = build_processes(data_pool=data_pool,
                                local_rank=local_rank,
                                **args_processes)
    for process in processes:
        process.load(root=root, iter=args_train['load_iter'])
    torch.cuda.empty_cache()

    for _ in range(data_pool['tot_iter']):
        for process in processes:
            if hasattr(process, "sch"):
                process.sch.step()

    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))
    use_amp = args_train.get("amp", False)

    val_inter = args_train['valid_interval']
    for iter in range(data_pool['tot_iter'] + 1, args_train['max_iter'] + 1):
        if args_train['max_iter'] - iter < 5 * val_inter:
            args_train['valid_interval'] = val_inter // 5
            args_train['snapshot_interval'] = val_inter // 5

        data_pool['state'] = "train"
        data_pool['tot_iter'] = iter
        for name, opt in data_pool['opt'].items():
            opt.zero_grad()

        with autocast(enabled=use_amp):
            for process in processes:
                process.run()

            tot_loss = 0.
            for name, (loss, r) in data_pool['loss'].items():
                if local_rank == 0:
                    writer.add_scalar("train/loss_{}".format(name), loss,
                                    data_pool['tot_iter'])
                if r > 0.001:
                    tot_loss += loss * r
        tot_loss.backward()
        for name, opt in data_pool['opt'].items():
            opt.step()
        for name, sch in data_pool['sch'].items():
            sch.step()

        if data_pool['tot_iter'] % args_train[
                'show_interval'] == 0 and local_rank == 0:
            to_log("iter: {}, epoch: {}, batch: {}/{}, loss: {}".format(
                data_pool['tot_iter'], data_pool['epoch'], data_pool['batch'],
                data_pool['train_batch'], tot_loss))
            writer.add_scalar("train/loss", tot_loss, data_pool['tot_iter'])

            for name, opt in data_pool['opt'].items():
                writer.add_scalar("train/lr_{}".format(name),
                                  opt.param_groups[0]['lr'],
                                  data_pool['tot_iter'])
        if data_pool['tot_iter'] % args_train['snapshot_interval'] == 0:
            for process in processes:
                process.save(root=root)

        if data_pool['tot_iter'] % args_train[
                'valid_interval'] == 0:  # and local_rank == 0:
            with torch.no_grad():
                data_pool['state'] = "valid"
                for val_iter in tqdm(range(data_pool['valid_batch'])):
                    for process in processes:
                        process.run()
            torch.cuda.empty_cache()

            if local_rank == 0:
                tot_loss = 0.
                for name, (loss, r) in data_pool['loss'].items():
                    tot_loss += loss * r
                    to_log("Loss: {}: {:.5f}".format(name, loss))
                    writer.add_scalar("valid/loss/{}".format(name),
                                    loss, data_pool['tot_iter'])
                for name, (loss, r) in data_pool['metric'].items():
                    to_log("Metric: {}: {:.5f}".format(name, loss))
                    writer.add_scalar("valid/{}".format(name),
                                    loss, data_pool['tot_iter'])
                writer.add_scalar("valid/loss", tot_loss, data_pool['tot_iter'])

    writer.close()


if __name__ == "__main__":
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # torch.multiprocessing.set_start_method('spawn')

    local_rank = int(os.environ["LOCAL_RANK"])
    assert torch.cuda.device_count() > local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            timeout=datetime.timedelta(seconds=36000))  # distributed backend
    device = torch.device('cuda', local_rank)
    world_size = dist.get_world_size()

    train(open_config(args.root), args.root)
