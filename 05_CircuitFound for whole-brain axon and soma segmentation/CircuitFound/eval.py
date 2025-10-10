import os
import warnings

import torch
import torch.multiprocessing as mp
from lib.models import EvalNet
import sys
sys.path.append('lib/')

from lib.utils import set_seed, dist_setup, get_conf



if __name__ == '__main__':
    args = get_conf()

    args.test = True

    # set seed if required
    set_seed(args.seed)

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    eval_model = EvalNet(args)
    # eval_model.eval_my_net()

    imgs = eval_model.test_3D_volume()

    for img in imgs:
        eval_model.segment_brain_batch(img)











