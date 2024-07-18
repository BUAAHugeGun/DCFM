from .processes import *
from .video_test import *
from tools.to_log import to_log

name2process = {
    "data": data_process,
    "model": model_process,
    "metric": metric_process,
    "test_data": video_test_data_process,
    "test_model": video_test_model_process,
    "test_metric": video_test_metric_process,
    "video_save": video_test_save_process,
}


def build_processes(data_pool, local_rank, **kwargs):
    ret = []
    for name, args_p in kwargs.items():
        type_name = args_p['process']
        to_log("building a {} process, args: \n{}".format(type_name, args_p))
        ret.append(name2process[type_name](data_pool,
                                      local_rank=local_rank,
                                      process_name = name,
                                      **args_p))
    return ret


if __name__ == "__main__":
    args = {
        "data": {
            "input_name": [],
            "output_name": ["x"],
            "tag": "cifar10",
            "world_size": 4,
            "bs": 16,
            "num_workers": 8,
            "args": {
                "root": "/home/LAB/r-yangwangwang/data/",
            }
        },
    }
    data_pool = {}
    build_processes(data_pool, local_rank=0, **args)
