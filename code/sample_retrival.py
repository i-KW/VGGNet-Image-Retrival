# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import operator

from dataset import VggDataset,VggDataset1
from visualizer import Visualizer
import visdom
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #标准化至[-1,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])



def make_model():
    model = models.vgg16(pretrained=True).features[:28]    #选择使用vgg16网络，第一次需先下载vgg16网络  改成19则使用vgg19
    # model = models.vgg19(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    # print(model)
    model = model.eval()  # 必须有，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()          # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model


if __name__ == "__main__":
    model = make_model()


    dataset = VggDataset1(r'E:\project\imageretriver\data\input',transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    model.eval()  # 必须有，不然会影响特征提取结果

    dataset_dis = {}

    for step, tensor in enumerate(dataloader):
        # for tensor in enumerate(dataloader):
        global image_name
        image_name = tensor[1]
        tensor = tensor[0]

        # print('s t',step,tensor)
        tensor = tensor.cuda()
        result = model(Variable(tensor))
        result_npy = result.data.cpu().numpy()

        sample_npy = result_npy[0]


    
    for npy in os.listdir(r'E:\project\imageretriver\data\npy16'):    #尝试用vgg16和19分别提取了图片的特征
    # for npy in os.listdir(r'E:\project\imageretriver\data\npy19'):
        data_npy = np.load(r'E:/project/imageretriver/data/npy16-555/' + npy)  # data_npy:tensor
        # data_npy = np.load(r'E:\project\imageretriver\data\npy19/'+npy)   #data_npy:tensor
        filename = npy[:-4]

        dis_Euclidean = np.sqrt(np.sum(np.square(data_npy-sample_npy)))
        dis_Manhattan = np.sum(np.abs(data_npy-sample_npy))
        dis_Chebyshev = np.max(np.abs(data_npy-sample_npy))   #三种比较特征的方式，分别为欧氏、曼哈顿、切比雪夫距离
        
        dataset_dis[filename] = dis_Chebyshev
        

    
    name = 'E:/project/imageretriver/data/result/' + image_name[0] + 'Chebyshev.txt'
    txt_result = open(name,'w+')
    print(sorted(dataset_dis.items(), key=operator.itemgetter(1)), file=txt_result) #将print的结果写入txt
    txt_result.close()

    result_list = sorted(dataset_dis.items(), key=operator.itemgetter(1))  #排序特征的距离，即相似度
    show_result = list(result_list[:24])       #取出前24个结果进行显示
    show_list = []
    show_filename = []
  
    for i in range(len(show_result)):
        show_list.append(show_result[i][0])

    
    for i in range(len(show_list)):
        show_filename.append(r'E:/project/imageretriver/data/IRSR/'+show_list[i].replace('-','/',1)+'.jpg') #修改名字为图片文件名

   
    visualizer = Visualizer(show_filename)
    visualizer.show_multi(show_filename) 



