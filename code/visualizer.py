import os
import matplotlib
import matplotlib.pyplot as plt
import visdom
import cv2
from pylab import *
class Visualizer():
    def __init__(self, file_list):
        self.file_list = file_list
        

    def show_multi(self, file_list):
        img = []
       
        for i in range(len(file_list)):
        
            try:
                imgread = cv2.imread(file_list[i])
                imgread = cv2.resize(imgread,(224,224),interpolation=cv2.INTER_AREA)
            except:
                imgread = cv2.imread(file_list[i-1])
                imgread = cv2.resize(imgread,(224,224),interpolation=cv2.INTER_AREA)
            img.append(imgread)
            
        htitch1 = np.hstack((img[0], img[1], img[2],img[3],img[4],img[5],img[6],img[7]))
        htitch2 = np.hstack((img[8], img[9], img[10], img[11], img[12], img[13], img[14], img[15]))
        htitch3 = np.hstack((img[16], img[17], img[18], img[19], img[20], img[21], img[22], img[23]))
        
        cv2.imshow("1-8", htitch1)
        cv2.imshow("9-16", htitch2)
        cv2.imshow("17-24", htitch3)
        

        cv2.waitKey(0) #使显示的图片停留，等待按键再关闭
        cv2.destroyAllWindows()



        






