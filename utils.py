import os
import shutil
import logging
from datetime import datetime

import torch
import torch.distributed as dist

from parser import args


def init():
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists(os.path.join('runs', args.task)):
        os.makedirs(os.path.join('runs', args.task))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name)):
        os.makedirs(os.path.join('runs', args.task, args.proj_name))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name, args.exp_name)):
        os.makedirs(os.path.join('runs', args.task, args.proj_name, args.exp_name))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files')):
        os.makedirs(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    if not os.path.exists(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'weights')):
        os.makedirs(os.path.join('runs', args.task, args.proj_name, args.exp_name, 'weights'))

    shutil.copy(args.main_program, os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    shutil.copy(f'model/{args.model_name}', os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    shutil.copy('utils.py', os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    shutil.copy(args.shell_name, os.path.join('runs', args.task, args.proj_name, args.exp_name, 'files'))
    

def cuda_seed_setup():
    # If you are working with a multi-GPU model, `torch.cuda.manual_seed()` is insufficient 
    # to get determinism. To seed all GPUs, use manual_seed_all().
    torch.cuda.manual_seed_all(args.seed)


def dist_setup(rank):
    # initialization for distributed training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)


def dist_cleanup():
    dist.destroy_process_group()


class Logger(object):
    def __init__(self, logger_name='Test', log_level=logging.INFO, log_path='runs', log_file='test.log'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_path, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def write(self, msg, rank=-1):
        if rank == 0:
            self.logger.info(msg)


def get_logger():
    logger_name = args.proj_name
    log_path = os.path.join('runs', args.task, args.proj_name, args.exp_name)
    log_file = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    return Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)
    