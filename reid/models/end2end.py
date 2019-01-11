from __future__ import absolute_import

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *


__all__ = ["End2End_AvgPooling"]


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, num_classes, is_output_feature, embeding_fea_size=1024, dropout=0.5, classifier="CrossEntropyLoss"):
        super(self.__class__, self).__init__()

        self.is_output_feature = is_output_feature

        # embeding
        self.embeding_fea_size = embeding_fea_size

        IDE_fea_size = 1024
        Ex_fea_size = 2048
        self.IDE_embeding = nn.Linear(input_feature_size, IDE_fea_size)
        self.IDE_embeding_bn = nn.BatchNorm1d(IDE_fea_size)
        self.Ex_embeding = nn.Linear(input_feature_size, Ex_fea_size)
        self.Ex_embeding_bn = nn.BatchNorm1d(Ex_fea_size)

        init.kaiming_normal_(self.IDE_embeding.weight, mode='fan_out')
        init.constant_(self.IDE_embeding.bias, 0)
        init.constant_(self.IDE_embeding_bn.weight, 1)
        init.constant_(self.IDE_embeding_bn.bias, 0)

        init.kaiming_normal_(self.Ex_embeding.weight, mode='fan_out')
        init.constant_(self.Ex_embeding.bias, 0)
        init.constant_(self.Ex_embeding_bn.weight, 1)        
        init.constant_(self.Ex_embeding_bn.bias, 0)

        self.drop = nn.Dropout(dropout)

        self.classify_fc = nn.Linear(IDE_fea_size, num_classes, bias=True)
        init.normal_(self.classify_fc.weight, std = 0.001)
        init.constant_(self.classify_fc.bias, 0)

        self.cls = classifier


    def forward(self, inputs):
        pool5 = inputs.mean(dim = 1) 

        if (not self.training)  and self.is_output_feature:
            return F.normalize(pool5, p=2, dim=1)

        """ IDE """
        # embeding
        net = self.drop(pool5)
        net = self.IDE_embeding(net)
        net = self.IDE_embeding_bn(net)
        net = F.relu(net)
        net = self.drop(net)
        # classifier
        predict = self.classify_fc(net)        

        if (not self.training)  and (not self.is_output_feature):
            return predict

        """ Exclusive """
        net = self.Ex_embeding(pool5)        
        net = self.Ex_embeding_bn(net)
        net = F.normalize(net, p=2, dim=1)
        Ex_feat = self.drop(net)        
        
        return predict, Ex_feat
        




class End2End_AvgPooling(nn.Module):

    def __init__(self, pretrained=True, dropout=0, num_classes=0, is_output_feature=True, embeding_fea_size=1024, classifier="CrossEntropyLoss", fixed_layer=True):
        super(self.__class__, self).__init__()

        self.CNN = resnet50(dropout=dropout, fixed_layer=fixed_layer)
        self.avg_pooling = AvgPooling(input_feature_size=2048, num_classes=num_classes, dropout=dropout, is_output_feature=is_output_feature, classifier=classifier,
                                      embeding_fea_size = embeding_fea_size)

    def forward(self, x):
        assert len(x.data.shape) == 5
        # reshape (batch, samples, ...) ==> (batch * samples, ...)
        oriShape = x.data.shape
        x = x.view(-1, oriShape[2], oriShape[3], oriShape[4])
        
        # resnet encoding
        resnet_feature = self.CNN(x)

        # reshape back into (batch, samples, ...)
        resnet_feature = resnet_feature.view(oriShape[0], oriShape[1], -1)

        # avg pooling
        # if eval and cut_off_before_logits, return predict;  else return avg pooling feature
        predict = self.avg_pooling(resnet_feature)
        return predict




