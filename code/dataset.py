# -*- coding: utf-8 -*-
#数据处理

import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
] #判断特定类型的图片文件


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

transform=transforms.Compose([
    transforms.Resize((224,224)), #缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #标准化至[-1,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ]
)

#定义自己的数据集合
class VggDataset(data.Dataset):

    def __init__(self,root,transform):
        self.imgs = []
        try:
            for filename in os.listdir(root):
                imgs = os.listdir(str(r'E:/project/imageretriver/data/IRSR/' + filename))
            
                for k in imgs:
                    if is_image_file(k):
                        self.imgs.append(str(r'E:/project/imageretriver/data/IRSR/' + filename +'/'+ str(k)))

        except:
            print('!')
        
        self.transforms=transform

    def __getitem__(self, index):
        img_path=self.imgs[index]
        # print(img_path,index)
        img_name = img_path[35:-4]
        img_name = img_name.replace(r'/','-')
        pil_img=Image.open(str(img_path))
        if self.transforms:
            try:
                global data
                data=self.transforms(pil_img) 
            except:
                print('!')
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)
        
        return data,img_name #will return a list of two element
        

    def __len__(self):
        return len(self.imgs)


class VggDataset1(data.Dataset):

    def __init__(self,root,transform):
        
        imgs = os.listdir(root)  
        try:
            self.imgs = [os.path.join(root, k) for k in imgs]
            self.transforms = transform
        except:
            print('!!')

    def __getitem__(self, index):
        img_path=self.imgs[index]

        img_name = img_path[35:-4]
        img_name = img_name.replace(r'/','-')

        pil_img=Image.open(str(img_path))
        if self.transforms:
            try:
                global data
                data=self.transforms(pil_img)   
            except:
                print('!!')
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)

        return data,img_name


    def __len__(self):
        return len(self.imgs)