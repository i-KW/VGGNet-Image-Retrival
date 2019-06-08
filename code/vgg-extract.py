# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


from dataset import VggDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #标准化至[-1,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])




def make_model():
    model = models.vgg16(pretrained=True).features[:28]  # 定位到第28层

    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()          # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model







if __name__ == "__main__":
    model = make_model()

    datapath = 'E:\project\imageretriver\data\IRSR'

    dataset = VggDataset(r'E:\project\imageretriver\data\IRSR',transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model.eval()  # 必须要有，不然会影响特征提取结果




    for epoch in range(3):  #

            for step, tensor in enumerate(dataloader):


                image_name = tensor[1]
                tensor = tensor[0]


                tensor = tensor.cuda()
                result = model(Variable(tensor))
                print(result.size())
                result_npy = result.data.cpu().numpy()

                np.save(str(r'E:/project/imageretriver/data/npy/'+image_name[0]+'.npy'),result_npy[0])


