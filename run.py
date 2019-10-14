from my_reid.eug import *
from my_reid import datasets
from my_reid import models
import numpy as np
import torch
import argparse
import os

from my_reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from my_reid.utils.serialization import load_checkpoint
from torch import nn
import time
import pickle


def resume(args):
    import re
    pattern=re.compile(r'step_(\d+)\.ckpt')
    start_step = -1
    ckpt_file = ""

    # find start step
    files = os.listdir(args.logs_dir)
    files.sort()
    for filename in files:
        try:
            iter_ = int(pattern.search(filename).groups()[0])
            print(iter_)
            if iter_ > start_step:
                start_step = iter_
                ckpt_file = osp.join(args.logs_dir, filename)
        except:
            continue

    # if need resume
    if start_step >= 0:
        print("continued from iter step", start_step)
    else:
        print("resume failed", start_step, files)
    return start_step, ckpt_file





def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir
    total_step = 100//args.EF + 1
    sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ str(args.EF)+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))

    # get all the labeled and unlabeled data for training

    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    num_all_examples = len(dataset_all.train)
    l_data, u_data = get_init_shot_in_cam1(dataset_all, load_path="./examples/{}_init_{}.pickle".format(dataset_all.name, args.init), init=args.init)
    
    resume_step, ckpt_file = -1, ''
    if args.resume:
        resume_step, ckpt_file = resume(args) 

    # initial the EUG algorithm 
    eug = EUG(batch_size=args.batch_size, num_classes=dataset_all.num_train_ids, 
            dataset=dataset_all, l_data=l_data, u_data=u_data, save_path=args.logs_dir, max_frames=args.max_frames,
            embeding_fea_size=args.fea, momentum=args.momentum, lamda=args.lamda)


    new_train_data = l_data 
    unselected_data = u_data
    for step in range(total_step):
        # for resume
        if step < resume_step: 
            continue
    
        ratio =  (step+1) * args.EF / 100
        nums_to_select = int(len(u_data) * ratio)
        if nums_to_select >= len(u_data):
                break

        print("Runing: EF={}%, step {}:\t Nums_to_be_select {} \t Ritio \t Logs-dir {}".format(
                args.EF, step,  nums_to_select, ratio, save_path))

        # train the model or load ckpt        
        eug.train(new_train_data, unselected_data, step, loss=args.loss, epochs=args.epochs, step_size=args.step_size, init_lr=0.1) if step != resume_step else eug.resume(ckpt_file, step)

        # evaluate 
        eug.evaluate(dataset_all.query, dataset_all.gallery)

        # pseudo-label and confidence score
        pred_y, pred_score = eug.estimate_label()

        # select data
        selected_idx = eug.select_top_data(pred_score, nums_to_select)

        # add new data
        new_train_data, unselected_data = eug.generate_new_train_data(selected_idx, pred_y)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Learning for One-Example re-ID')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)  
    parser.add_argument('-f', '--fea', type=int, default=1024)
    parser.add_argument('--EF', type=int, default=10)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=900)
    parser.add_argument('--loss', type=str, default='ExLoss', choices=['CrossEntropyLoss', 'ExLoss'])
    parser.add_argument('--init', type=float, default=-1)
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-s', '--step_size', type=int, default=55)
    parser.add_argument('--lamda', type=float, default=0.5)
    main(parser.parse_args())

