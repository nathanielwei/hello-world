#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:21:12 2018

@author: weizikai
"""

import argparse
import os
import shutil
import time
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pdb

from models.model_lstmfc3net import LSTMFC3Net
from dataloaders.dataloader_bardatasetrw import BarDatasetRW
'''
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
'''
parser = argparse.ArgumentParser(description='PyTorch LSTMFC3Net Training')

'''
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
'''
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://10.1.75.35', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

parser.add_argument('--gpu-ids', type=int, nargs="+", default=[0],
                    help='gpu ids')
parser.add_argument('--data-path', type=str, default='./data/RBL8.csv',
                    help='path for data')
parser.add_argument('--save-path', type=str, default='./checkpoints',
                    help='path for saving models')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    
    
    model = LSTMFC3Net()   

    model = torch.nn.DataParallel(model, device_ids = args.gpu_ids)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.NLLLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    
    filename = args.data_path
    train_dataset= BarDatasetRW(filename, rws = 10, train_phase = True, train_ratio = 0.66)
    val_dataset= BarDatasetRW(filename, rws = 10, train_phase = False, train_ratio = 0.66)
    
    #train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 2, drop_last=True)
    #val_loader = DataLoader(val_dataset, batch_size = 256, shuffle = False, num_workers = 2, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True)
    
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch % 10 == 0:
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_all = AverageMeter()
    acc_0 = AverageMeter()
    acc_1 = AverageMeter()
    acc_2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        input = data['bars']
        target = data['labels']
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.long().cuda()
        input = Variable(input)
        target = Variable(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec, flag = accuracy(output.cpu().data, target.cpu().data)
        losses.update(loss.cpu().data[0], input.size(0))
        acc_all.update(prec[0], input.size(0))
        if flag[0]==1:
            acc_0.update(prec[1], input.size(0))
        if flag[1]==1:
            acc_1.update(prec[2], input.size(0))
        if flag[2]==1:
            acc_2.update(prec[3], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@all {acc_all.val:.3f} ({acc_all.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc_all=acc_all))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_all = AverageMeter()
    acc_0 = AverageMeter()
    acc_1 = AverageMeter()
    acc_2 = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        # measure data loading time
        input = data['bars']
        target = data['labels']
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.long().cuda()
        input = Variable(input, volatile=True)
        target = Variable(target, volatile=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec, flag = accuracy(output.cpu().data, target.cpu().data)
        losses.update(loss.cpu().data[0], input.size(0))
        acc_all.update(prec[0], input.size(0))
        if flag[0]==1:
            acc_0.update(prec[1], input.size(0))
        if flag[1]==1:
            acc_1.update(prec[2], input.size(0))
        if flag[2]==1:
            acc_2.update(prec[3], input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            print('Test: \t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@all {acc_all.val:.3f} ({acc_all.avg:.3f})'.format(
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, acc_all=acc_all))

    print(' * Epoch: [{0}] Prec@all {acc_all.avg:.3f}'
          .format(epoch, acc_all=acc_all))

    return acc_all.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_file = osp.join(args.save_path, filename)
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, osp.join(args.save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, labels=3):
    """Computes the precision@k for the specified values of k"""

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / target.size(0))[0])
    
    
    flag = []
    for label in range(labels):
        if (target==label).sum()==0:
            res.append(0)
            flag.append(0)
            continue
        flag.append(1)
        results_k = torch.index_select(correct,1,(target==label).nonzero().view(-1))
        correct_k = results_k.view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / results_k.size(1))[0])
    
    return res, flag


if __name__ == '__main__':
    main()
    