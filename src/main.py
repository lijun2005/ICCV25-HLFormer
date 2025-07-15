import os
import argparse
import numpy as np
import random
import sys
import time
from tqdm import tqdm
import ipdb
import pickle

import torch
import torch.nn as nn

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Opts.builder import get_opts
from Losses.builder import get_losses
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Partially Relevant Video Retrieval")
parser.add_argument(
    '-d', '--dataset_name', default='tvr', type=str, metavar='DATASET', help='dataset name', 
    choices=['tvr', 'act', 'cha']
)
parser.add_argument(
    '--gpu', default = '0', type = str, help = 'specify gpu device'
    )
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()


def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer):

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()

    model.train()

    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                    unit="batch", dynamic_ncols=True)

    for idx, batch in enumerate(train_bar):

        batch = gpu(batch)

        optimizer.zero_grad()

        input_list = model(batch)

        loss = criterion(input_list, batch)
        
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.cpu().item())

        train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(cfg['model_name'], epoch, idx, loss))

    return loss_meter.avg


def val_one_epoch(epoch, context_dataloader, query_eval_loader, model, val_criterion, cfg, optimizer, best_val, loss_meter, logger):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)

    if val_meter[4] > best_val[4]:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'
                
    logger.info('==========================================================================================================')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('Best: R@1: {:.1f} R@5: {:.1f} R@10: {:.1f} R@100: {:.1f} Rsum: {:.1f}'.format(best_val[0], best_val[1], best_val[2], best_val[3], best_val[4]))
    logger.info('==========================================================================================================')
        
    return val_meter, best_val, es


def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    
    logger.info('==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('==========================================================================================================')


def main():
    cfg = get_configs(args.dataset_name)

    # set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info('Partially Relevant Video Retrieval Training: {}'.format(cfg['dataset_name']))

    # set seed
    set_seed(cfg['seed'])
    logger.info('set seed: {}'.format(cfg['seed']))

    # hyper parameter
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    logger.info('Hyper Parameter ......')
    logger.info(cfg)

    # dataset
    logger.info('Loading Data ......')
    cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = get_datasets(cfg)

    # model
    logger.info('Loading Model ......') 
    model = get_models(cfg)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"Total Parameters: {total_params:.2f}M")
    print(f"Trainable Parameters: {trainable_params:.2f}M")
    
    # initial
    current_epoch = -1
    es_cnt = 0
    best_val = [0., 0., 0., 0., 0.]
    if args.resume != '':
        logger.info('Resume from {}'.format(args.resume))
        _, model_state_dict, optimizer_state_dict, current_epoch, best_val = load_ckpt(args.resume)
        model.load_state_dict(model_state_dict)
    model = model.cuda()
    if len(device_ids) > 1:
        model = nn.DataParallel(model)
    
    criterion = get_losses(cfg)
    val_criterion = get_validations(cfg)

    if args.eval:
        if args.resume == '':
            logger.info('No trained ckpt load !!!') 
        else:
            with torch.no_grad():
                validation(test_context_dataloader, test_query_eval_loader, model, val_criterion, cfg, logger, args.resume)
        exit(0)

    optimizer = get_opts(cfg, model, train_loader)
    if args.resume != '':
        optimizer.load_state_dict(optimizer_state_dict)

    for epoch in range(current_epoch + 1, cfg['n_epoch']):

        ############## train
        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer)

        ############## val
        with torch.no_grad():
            val_meter, best_val, es = val_one_epoch(epoch, context_dataloader, query_eval_loader, model, 
                    val_criterion, cfg, optimizer, best_val, loss_meter, logger)

        ############## early stop
        if not es:
            es_cnt = 0
        else:
            es_cnt += 1
            if cfg['max_es_cnt'] != -1 and es_cnt > cfg['max_es_cnt']:  # early stop
                logger.info('Early Stop !!!') 
                exit(0)


if __name__ == '__main__':
    main()