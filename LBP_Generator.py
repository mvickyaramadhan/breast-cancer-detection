#!/usr/bin/env python
# -- coding: utf-8 --
from config import opt
import os
import models
#import face_alignment
#from skimage import io
from torch.autograd import Variable
from torchnet import meter
from utils import Visualizer
from tqdm import tqdm
from torchvision import transforms
import torchvision
import torch
from torchsummary import summary
import json
import numpy as np
import cv2

class DataHandle():

    def __init__(self,scale=2.7,image_size=224,use_gpu=False,transform=None,data_source = None):
        self.transform = transform
        self.scale = scale
        self.image_size = image_size

    def thresholded(self, center, pixels):
        out = []
        for a in pixels:
            if a >= center:
                out.append(1)
            else:
                out.append(0)
        return out

    def get_pixel_else_0(self, l, idx, idy, default=0):
        try:
            return l[idx,idy]
        except IndexError:
            return default

    def get_data(self,image_path):#第二步装载数据，返回[img,label]
        img = cv2.imread(image_path)
        print(image_path)
        #cv2.imwrite('data_ori.jpg',img)
        img_test = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #cv2.imwrite('data_yuv.jpg',img_test)
        y1, u1, v1 = cv2.split(img_test)
        #cv2.imwrite('data_channel.jpg',y1)
        transformed_img = v1.copy()
        for x in range(0, len(v1)):
            for y in range(0, len(v1[0])):
                center        = v1[x,y]
                top_left      = self.get_pixel_else_0(v1, x-1, y-1)
                top_up        = self.get_pixel_else_0(v1, x, y-1)
                top_right     = self.get_pixel_else_0(v1, x+1, y-1)
                right         = self.get_pixel_else_0(v1, x+1, y )
                left          = self.get_pixel_else_0(v1, x-1, y )
                bottom_left   = self.get_pixel_else_0(v1, x-1, y+1)
                bottom_right  = self.get_pixel_else_0(v1, x+1, y+1)
                bottom_down   = self.get_pixel_else_0(v1, x,   y+1 )

                values = self.thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                            bottom_down, bottom_left, left])

                weights = [1, 2, 4, 8, 16, 32, 64, 128]
                res = 0
                for a in range(0, len(values)):
                    res += weights[a] * values[a]

                transformed_img.itemset((x,y), res)

            #print (x)

        cv2.imwrite(image_path, transformed_img)
        #cv2.imshow('data.jpg',img)
        #cv2.waitKey()
        if 0:
            cv2.imshow('crop face',img)
            cv2.waitKey(0)

        return np.transpose(np.array(img, dtype = np.float32), (2, 0, 1)), image_path

    def __len__(self):
        return len(self.img_label)
    

def LBP_Generator(**kwargs):
    import glob
    images = glob.glob(kwargs['images'])
    assert len(images)>0
    data_handle = DataHandle(
                        scale = opt.cropscale,
                        use_gpu = opt.use_gpu,
			transform = None,
			data_source='none')
    opt.parse(kwargs)
    tqbar = tqdm(enumerate(images),desc='LBP Generator')
    for idx,imgdir in tqbar:
        data,_ = data_handle.get_data(imgdir)

if __name__=='__main__':
    import fire
    fire.Fire()