# -*- coding: utf-8 -*-
import numpy as np
import cv2
#from frames2video import pic2video
import warnings
import torch
import numpy as np
import argparse
import time
from models import *  # the path is depended on where you create this module

from utils.flow_utils import flow2img

from PIL import Image
import os
import cv2

dirvideo = 'D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\'
dirvideo = "D:\Study\Datasets\AATEST\latest\\video\\part1\\"
# dirvideo = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\HAMLYN\\"
# dirvideo = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\HAMLYN\\"
listinit = os.listdir(dirvideo)
list = listinit.copy()
new_size = (512,384)
for i in range(0, len(list)):
    path = dirvideo+os.path.join(list[i])
    if path.endswith('.mp4'):#or path.endswith('avi'):
        print(path)
        cap = cv2.VideoCapture(path)
        _, frame1 = cap.read()

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        out = cv2.VideoWriter(path[:-4]+"_"+'.avi', fourcc, 30, new_size)


        while (i):

            # Capture another frame and convert to gray scale
            ret, frame2 = cap.read()
            if ret != True:
                break

            b = cv2.resize(frame2, new_size,interpolation=cv2.INTER_LINEAR)
            out.write(b)



        cap.release()
        # out.release()
        cv2.destroyAllWindows()











