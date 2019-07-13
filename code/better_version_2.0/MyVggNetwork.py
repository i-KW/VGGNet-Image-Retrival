# -*- coding: utf-8 -*-

import os
import torch.nn as nn
import torch
import torchvision.models as models



class MyVGG(nn.Module):


    def __init__(self):
        super(MyVGG, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.features = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self.model.classifier[0:4]


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
