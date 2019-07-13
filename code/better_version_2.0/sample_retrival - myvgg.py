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
from MyVggNetwork import MyVGG
from PIL import Image
import cv2
import operator
from MyVggNetwork import MyVGG

from dataset2 import VggDataset,VggDataset1
from visualizer import Visualizer
import visdom
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #标准化至[-1,1]

])


if __name__ == "__main__":

    dataset = VggDataset1(r'E:\project\imageretriver\data\input',transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)


    dataset_dis = {}

    for step, tensor in enumerate(dataloader):

        global image_name
        image_name = tensor[1]
        tensor = tensor[0]




        tensor = tensor.cuda()



        model = MyVGG()

        model = model.eval()
        model.cuda()
        result = model(Variable(tensor))


        result_npy = result.data.cpu().numpy()

        sample_npy = result_npy[0]



    for npy in os.listdir(r'E:\project\imageretriver\data\npy-myvgg'):


        data_npy = np.load(r'E:/project/imageretriver/data/npy-myvgg/' + npy)  # data_npy:tensor

        filename = npy[:-4]


        dis_Euclidean = np.sqrt(np.sum(np.square(data_npy-sample_npy)))
        dis_Manhattan = np.sum(np.abs(data_npy-sample_npy))
        dis_Chebyshev = np.max(np.abs(data_npy-sample_npy))



        dataset_dis[filename] = dis_Euclidean #choose Euclidean distance



    name = 'E:/project/imageretriver/data/result/' + image_name[0] + 'Chebyshev.txt'

    txt_result = open(name,'w+')

    print(sorted(dataset_dis.items(), key=operator.itemgetter(1)), file=txt_result)
    txt_result.close()

    result_list = sorted(dataset_dis.items(), key=operator.itemgetter(1))

    show_result = list(result_list[:24])

    show_list = []
    show_filename = []

    for i in range(len(show_result)):
        show_list.append(show_result[i][0])


    for i in range(len(show_list)):
        show_filename.append(r'E:/project/imageretriver/data/IRSR/'+show_list[i].replace('-','/',1)+'.jpg')

  
    visualizer = Visualizer(show_filename)
    visualizer.show_multi(show_filename)



