from JUST4PAIR import just4pair
import os
import time
import os
from warp import *
from pyoptflow import HornSchunck, getimgfiles
import torch
import numpy as np
import argparse
from utils.flow_utils import flow2img
import cv2
from warp import *

from models import FlowNet2




currentpath = 'D:\Study\Datasets\hamlyn\current\\'
nextpath = 'D:\Study\Datasets\hamlyn\\next\\'
flowpath = 'D:\Study\Datasets\hamlyn\\'
esti_curr = 'D:\Study\Datasets\hamlyn\\'
error = 'D:\Study\Datasets\hamlyn\\'


currentpath = 'D:\Study\Datasets\cholec80\current\\'
nextpath = 'D:\Study\Datasets\cholec80\\next\\'
flowpath = 'D:\Study\Datasets\cholec80\\'
esti_curr = 'D:\Study\Datasets\cholec80\\'
error = 'D:\Study\Datasets\cholec80\\'


names = []
for f_name in os.listdir(currentpath):
    if f_name.endswith('.png'):
        names.append(f_name)


for name in names:
    # if int(name[5:7])>11:
        just4pair(name,currentpath,nextpath,flowpath,esti_curr,error,'HS')


#GF  HS NN    IN TERMS OF EFFECIENCY
#NN  GF HS    IN TERMS OF GOODNESS