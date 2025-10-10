import argparse
import os
import os.path
import random

import numpy as np
import torch


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/stateNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("state")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_opt_checkpoint_path(path):
    basedir = os.path.dirname(path)
    resume_step = parse_resume_step_from_filename(path)
    filename = f"opt{resume_step:06d}.pt"
    opt_path = os.path.join(basedir, filename)
    return opt_path


def set_random_seed(seed=0):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False