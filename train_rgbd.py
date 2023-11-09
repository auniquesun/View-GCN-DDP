import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.Trainer import Trainer
from tools.ImgDataset import RGBD_MultiView, RGBD_SingleView
from model.view_gcn import view_GCN, SVCNN

from utils import init, get_logger, dist_setup, cuda_seed_setup, dist_cleanup
from parser import args


def entry(rank, num_devices):

    dist_setup(rank)

    cuda_seed_setup()

    assert args.batch_size % args.world_size == args.test_batch_size % args.world_size == 0, \
        'Argument `batch_size` and `test_batch_size` should be divisible by `world_size`'

    logger = get_logger()
    logger.write(str(args), rank=rank)
    msg = f'{num_devices} GPUs are available and {args.world_size} of them are used. Ready for DDP training ...'
    logger.write(msg, rank=rank)

    if args.stage_one:
        # --- 1.1 prepare data
        sv_train_set = RGBD_SingleView(args.train_path, trial_id=args.trial_id, test_mode=False, num_classes=args.num_obj_classes)
        sv_samples_per_gpu = args.base_model_batch_size // args.world_size
        sv_train_sampler = DistributedSampler(sv_train_set, num_replicas=args.world_size, rank=rank)
        sv_train_loader = DataLoader(sv_train_set, sampler=sv_train_sampler, batch_size=sv_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(sv_train_loader): {len(sv_train_loader)}', rank=rank)

        sv_test_set = RGBD_SingleView(args.test_path, trial_id=args.trial_id, test_mode=True, num_classes=args.num_obj_classes)
        sv_test_samples_per_gpu = args.base_model_test_batch_size // args.world_size
        sv_test_sampler = DistributedSampler(sv_test_set, num_replicas=args.world_size, rank=rank)
        sv_test_loader = DataLoader(sv_test_set, sampler=sv_test_sampler, batch_size=sv_test_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(sv_test_loader): {len(sv_test_loader)}', rank=rank)

        # --- 1.2 define model
        cnet = SVCNN(args.exp_name, nclasses=args.num_obj_classes, pretraining=True, cnn_name=args.base_model_name).to(rank)

        # --- 1.3 wrap model with DDP
        cnet_ddp = DDP(cnet, device_ids=[rank], find_unused_parameters=True)

        # --- 1.4 construct optimizer
        optimizer = optim.SGD(cnet_ddp.parameters(), lr=1e-2, weight_decay=args.weight_decay, momentum=0.9)

        # --- 1.5 define loss
        sv_criterion = nn.CrossEntropyLoss()

        # --- 1.6 build trainer and start training
        sv_trainer = Trainer(cnet_ddp, sv_train_loader, sv_train_sampler, sv_test_loader, sv_test_sampler, optimizer, 
                                sv_criterion, 'svcnn', type='sv')
        sv_trainer.train(rank, logger, args)

    if args.stage_two:
        # --- 2.1 prepare data
        mv_train_set = RGBD_MultiView(args.train_path, trial_id=args.trial_id, num_views=args.num_views, test_mode=False, num_classes=args.num_obj_classes)
        mv_samples_per_gpu = args.batch_size // args.world_size
        mv_train_sampler = DistributedSampler(mv_train_set, num_replicas=args.world_size, rank=rank)
        mv_train_loader = DataLoader(mv_train_set, sampler=mv_train_sampler, batch_size=mv_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(mv_train_loader): {len(mv_train_loader)}', rank=rank)
        
        mv_test_set = RGBD_MultiView(args.test_path, trial_id=args.trial_id, num_views=args.num_views, test_mode=True, num_classes=args.num_obj_classes)
        mv_test_samples_per_gpu = args.test_batch_size // args.world_size
        mv_test_sampler = DistributedSampler(mv_test_set, num_replicas=args.world_size, rank=rank)
        mv_test_loader = DataLoader(mv_test_set, sampler=mv_test_sampler, batch_size=mv_test_samples_per_gpu, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True,)
        logger.write(f'len(mv_test_loader): {len(mv_test_loader)}', rank=rank)

        # --- 2.2.1 construct base model and load pretrained-weights
        sv_classifier = SVCNN(args.exp_name, nclasses=args.num_obj_classes, pretraining=True, cnn_name=args.base_model_name).to(rank)
        if args.resume: #'You should specify the pretrained `base_model_weights`'
            logger.write(f'Loading pretrained weights of {args.base_model_name} on {args.dataset} ...', rank=rank)
            map_location = torch.device('cuda:%d' % rank)
            sv_pretrained = torch.load(args.base_model_weights, map_location=map_location)
            sv_classifier.load_state_dict(sv_pretrained, strict=True)
        else:
            logger.write(f'Constructing MultiViewTransformer without pretrained `sv_classifier.feature_extractor` ...', rank=rank)

        # --- 2.2.2 construct model
        viewgcn = view_GCN(args.exp_name, sv_classifier, nclasses=args.num_obj_classes, cnn_name=args.base_model_name, num_views=args.num_views).to(rank)

        # --- 2.3 wrap model with DDP
        viewgcn_ddp = DDP(viewgcn, device_ids=[rank], find_unused_parameters=False)

        # --- 2.4 construct optimizer
        optimizer = optim.SGD(viewgcn_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)

        # --- 2.5 define loss function
        mv_criterion = nn.CrossEntropyLoss()

        # --- 2.6 construct trainer and start training
        mv_trainer = Trainer(viewgcn_ddp, mv_train_loader, mv_train_sampler, mv_test_loader, mv_test_sampler, 
                                    optimizer, mv_criterion, 'view-gcn', type='mv')
        mv_trainer.train(rank, logger, args)

    dist_cleanup()


if __name__ == '__main__':
    init()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            mp.spawn(entry, args=(num_devices,), nprocs=args.world_size)
        else:
            sys.exit('Only one GPU is available, the process will be much slower. Exit')
    else:
        sys.exit('CUDA is unavailable! Exit')
    