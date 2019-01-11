from __future__ import print_function, absolute_import
import time

import torch
from torch import nn
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter



def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class BaseTrainer(object):
    def __init__(self, model, criterion, lamda, fixed_layer=True):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.ide_criterion = nn.CrossEntropyLoss().cuda()
        self.u_criterion = criterion
        self.fixed_layer = fixed_layer
        self.label_ratio = lamda

    def train(self, epoch, ide_data_loader, u_loader, optimizer, use_unselcted_data, print_freq=30):
        self.model.train()

        if self.fixed_layer:
            # The following code is used to keep the BN on the first three block fixed 
            fixed_bns = []
            for idx, (name, module) in enumerate(self.model.module.named_modules()):
                if name.find("layer3") != -1:
                    assert len(fixed_bns) == 22
                    break
                if name.find("bn") != -1:
                    fixed_bns.append(name)
                    module.eval() 


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()


        u_loader = iter(cycle(u_loader))
        trainlog = TrainingLog(epoch, print_freq, len(ide_data_loader))

        for i, ide_inputs in enumerate(ide_data_loader):
            data_time.update(time.time() - end)

            # ide forward
            ide_inputs, ide_targets = self._parse_data(ide_inputs, 'ide')
            ide_loss, ide_prec1 = self._forward(ide_inputs, ide_targets, 'ide')
            weighted_loss = ide_loss

            u_loss, u_prec1 = ide_loss, ide_prec1

            # unselcted part forward
            if use_unselcted_data:
                u_inputs = next(u_loader)
                u_inputs, u_targets = self._parse_data(u_inputs, 'u')
                u_loss, u_prec1 = self._forward(u_inputs, u_targets, 'u')
                weighted_loss = self.get_weighted_loss(ide_loss, u_loss)

            # update weighted loss and bp
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            trainlog.update(i, weighted_loss, ide_loss, u_loss, ide_prec1, u_prec1, ide_targets)



    def get_weighted_loss(self, ide_loss, u_loss):        
        weighted_loss = ide_loss * self.label_ratio + u_loss * (1-self.label_ratio)
        return weighted_loss 

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs, mode):
        imgs, _, pids, indexs, _ = inputs
        inputs = Variable(imgs, requires_grad=False)
        if mode == "u":
            targets = Variable(indexs.cuda())
        elif mode == "ide":
            targets = Variable(pids.cuda())  
        else:
            raise KeyError       
        return inputs, targets


    def _forward(self, inputs, targets, mode):
        ide_preds, u_feats = self.model(inputs)

        if mode == "ide":
            # id predictions
            ide_loss = self.ide_criterion(ide_preds, targets)
            ide_prec, = accuracy(ide_preds.data, targets.data)
            ide_prec = ide_prec[0]
            return ide_loss, ide_prec
        elif mode == 'u':
            # u predictions
            u_loss, outputs = self.u_criterion(u_feats, targets)
            u_prec, = accuracy(outputs.data, targets.data)
            u_prec = u_prec[0]
            return u_loss, u_prec
        else:
            raise KeyError


class TrainingLog():
    def __init__(self, epoch, print_freq, data_len):
        self.batch_time = AverageMeter()
        self.losses = AverageMeter()
        self.ide_losses = AverageMeter()
        self.u_losses = AverageMeter()
        self.ide_precisions = AverageMeter()
        self.u_precisions = AverageMeter()
        self.time = time.time()

        self.epoch = epoch
        self.print_freq = print_freq
        self.data_len = data_len

    def update(self, step, weighted_loss, ide_loss, u_loss, ide_prec, u_prec, targets):
        # update time
        t = time.time()
        self.batch_time.update(t - self.time)
        self.time = t

        # weighted loss
        self.losses.update(weighted_loss.item(), targets.size(0))
        self.ide_losses.update(ide_loss.item(), targets.size(0))
        self.u_losses.update(u_loss.item())
        
        # id precision
        self.ide_precisions.update(ide_prec, targets.size(0))
        self.u_precisions.update(u_prec, targets.size(0))

        if (step + 1) % self.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Loss {:.3f} ({:.3f})\t'
                  'IDE_Loss {:.3f} ({:.3f})\t'
                  'ExLoss {:.3f} ({:.3f})\t'
                  'IDE_Prec {:.1%} ({:.1%})\t'
                  'ExPrec {:.1%} ({:.1%})\t'
                  .format(self.epoch, step + 1, self.data_len,
                          self.batch_time.val, self.batch_time.avg,
                          self.losses.val, self.losses.avg,
                          self.ide_losses.val, self.ide_losses.avg,
                          self.u_losses.val, self.u_losses.avg,
                          self.ide_precisions.val, self.ide_precisions.avg,
                          self.u_precisions.val, self.u_precisions.avg))     
