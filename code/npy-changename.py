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





for file in os.listdir(r'E:\project\imageretriver\data\npy -copy'):
        filename = file
        print(filename)
        if filename[:7]=='traffic':
                newfilename = 'traffic sign/' + filename[8:]
        else:
                newfilename = filename.replace('_','-',1)
        print(newfilename)

        tensor = np.load('E:\project\imageretriver\data/npy -copy/' + filename )

        print(tensor)

        np.save(str('E:\project\imageretriver\data/npy16-555/' + newfilename ),tensor)