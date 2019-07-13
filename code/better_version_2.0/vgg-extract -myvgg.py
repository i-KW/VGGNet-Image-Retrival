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
from MyVggNetwork import MyVGG

from dataset2 import VggDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #标准化至[-1,1]

])



if __name__ == "__main__":

    datapath = 'E:\project\imageretriver\data\IRSR'

    dataset = VggDataset(r'E:\project\imageretriver\data\IRSR',transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)






    for epoch in range(3):

            for step, tensor in enumerate(dataloader):


                image_name = tensor[1]
                tensor = tensor[0]
                tensor = tensor.cuda()
                model = MyVGG()   # 使用改写的VGG模型 use the vgg net that I rewrite
                model = model.eval()
                model.cuda()
                result = model(Variable(tensor))
                print(result.size())
                result_npy = result.data.cpu().numpy()

                np.save(str(r'E:/project/imageretriver/data/npy-myvgg/'+image_name[0]+'.npy'),result_npy[0])


