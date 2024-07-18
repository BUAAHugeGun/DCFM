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


class base_process():

    def __init__(self, data_pool, local_rank, workplace) -> None:
        self.data_pool = data_pool
        self.local_rank = local_rank
        self.device = torch.device('cuda', local_rank)
        self.world_size = dist.get_world_size()
        self.input_name = []
        self.output_name = []
        self.lst_state = None
        self.state = None
        self.workplace = workplace

    def is_in_workplace(self, state:str):
        if len(self.workplace) > 0 and state not in self.workplace:
            return False
        return True

    def run(self, **kwargs):
        self.update_state()
        if not self.is_in_workplace(self.state):
            return
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

    def update_state(self):
        self.lst_state = self.state
        self.state = self.data_pool['state']


class data_process(base_process):

    def __init__(self, data_pool, local_rank, **kwargs) -> None:
        super(data_process, self).__init__(data_pool, local_rank,
                                           kwargs.get('workplace', []))
        # self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        dataset = getattr(sys.modules[__name__], kwargs['tag'])
        self.train_dataset = dataset.get_dataset(**kwargs['args'],
                                                 split="train")
        self.valid_dataset = dataset.get_dataset(**kwargs['args'],
                                                 split="valid")
        self.num_workers = kwargs['num_workers']
        self.bs = kwargs['bs'] // self.world_size
        self.bs_val = kwargs.get('bs_val', 1)

        self.train_loader, self.sampler = self.get_loader("train")
        self.valid_loarder, _ = self.get_loader("valid")
        if self.is_in_workplace("train"):
            self.data_pool['train_batch'] = len(self.train_loader)
        if self.is_in_workplace("valid"):
            self.data_pool['valid_batch'] = len(self.valid_loarder)
        self.iter_train = iter(self.train_loader)
        self.iter_valid = iter(self.valid_loarder)
        # print(len(self.train_dataset), self.data_pool['train_batch'])

    def get_loader(self, split):
        if split == "train":
            sampler = distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True)
            bs = self.bs
            nw = self.num_workers
            dataset = self.train_dataset
        else:
            bs = self.bs_val
            sampler = None
            nw = 0
            dataset = self.valid_dataset
        dataloader = DataLoader(dataset,
                                bs,
                                shuffle=False,
                                num_workers=nw,
                                sampler=sampler)
        return dataloader, sampler

    def load(self, root, iter):
        self.data_pool['tot_iter'] = max(iter, 0)

    def infer(self):
        if self.state == "train":
            epoch = (self.data_pool['tot_iter'] - 1) // self.data_pool['train_batch']
            # if self.data_pool['tot_iter'] % self.data_pool['train_batch'] == 1:
            if self.data_pool['tot_iter'] - epoch * self.data_pool['train_batch'] == 1:
                self.sampler.set_epoch(epoch)
                self.iter_train = iter(self.train_loader)
            self.data_pool['epoch'] = epoch
            self.data_pool['batch'] = (self.data_pool['tot_iter'] -
                                       1) % self.data_pool['train_batch'] + 1
            self.iter_data = self.iter_train
        elif self.state != self.lst_state:
            self.iter_valid = iter(self.valid_loarder)
            self.iter_data = self.iter_valid

        x = next(self.iter_data)
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, torch.Tensor):
                    v.to(self.device)
        elif isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    v.to(self.device)
        return x


class model_process(base_process):

    def __init__(self, data_pool, local_rank, **kwargs) -> None:
        super(model_process, self).__init__(data_pool, local_rank,
                                            kwargs.get('workplace', []))
        self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        self.model_name = kwargs['tag']
        self.ckpt_name = kwargs['process_name']
        self.model = getattr(sys.modules[__name__], self.model_name)
        self.model = self.model.get_model(**kwargs.get("args", {}))
        self.model = self.model.to(self.device)
        if kwargs['ddp']:
            self.model = DDP(self.model,
                             device_ids=[self.local_rank],
                             find_unused_parameters=True)
        to_log(get_parameter_number(self.model))
        if "opt" in kwargs:
            self.build_opt(**kwargs['opt'])
            self.data_pool['opt'][self.ckpt_name] = self.opt
        if "sch" in kwargs:
            self.build_sch(**kwargs['sch'])
            self.data_pool['sch'][self.ckpt_name] = self.sch

    def build_opt(self, tag, **args):
        self.opt = getattr(optim, tag)(params=self.model.parameters(), **args)

    def build_sch(self, tag, **args):
        if hasattr(optim.lr_scheduler, tag):
            self.sch = getattr(optim.lr_scheduler, tag)(optimizer=self.opt, **args)
        else:
            self.sch = getattr(tools.utils, tag)(optimizer=self.opt, **args)

    def load(self, root, iter):
        load_iter = load_ckpt({
            self.ckpt_name: self.model,
        }, iter, root)
        if load_iter is not None:
            self.data_pool['tot_iter'] = load_iter

    def save(self, root):
        path = os.path.join(
            root, "logs/{}_epoch_{}.pth".format(self.ckpt_name,
                                                self.data_pool['tot_iter']))
        torch.save(self.model.state_dict(), path)

    def infer(self):
        if self.state != self.lst_state:
            if self.state == "train":
                self.model.train()
            elif self.state != "train":
                self.model.eval()

        x = self.get_data()
        if len(x) == 1:
            y = self.model(*x)
        else:
            y = self.model(x)
        return y


class metric_process(base_process):

    def __init__(self, data_pool, local_rank, **kwargs) -> None:
        super(metric_process, self).__init__(data_pool, local_rank,
                                             kwargs.get('workplace', []))
        self.input_name = kwargs['input_name']
        self.output_name = kwargs['output_name']
        self.model_name = kwargs['tag']
        self.model = get_metric(self.model_name, **kwargs.get("args", {}))
        self.ratio = kwargs.get('lambda', 1.0)

    def infer(self):
        if self.state == "train":
            self.model.reset()
        if self.state != self.lst_state:
            if self.state != "train":
                self.model.reset()

        x = self.get_data()
        self.model(*x)
        y = self.model.get()
        # y = y * self.ratio
        return (y, self.ratio)