# -*- coding: utf-8 -*-
import numpy as np
import cv2
from frames2video import pic2video
import warnings
# warnings.filterwarnings("ignore")
import torch
import numpy as np
import argparse
import time
from models import FlowNet2  # the path is depended on where you create this module

from utils.flow_utils import flow2img
import matplotlib.pyplot as plt
import os
import shutil





# cap = cv2.VideoCapture('../Datasets/real_test/real_test.mp4')
cap = cv2.VideoCapture('../Datasets/hand_test/hand_test1.mp4')
# 获取第一帧
ret, frame1 = cap.read()
prvs = frame1
i = 1  # 控制实现的张数

save_path = '../Datasets/hand_test/hand_test_result/'

start = time.time()

# obtain the necessary args for construct the flownet framework
parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)

args = parser.parse_args()

# initial a Net
net = FlowNet2(args).cuda()
# load the state_dict
dict = torch.load("checkpoints/FlowNet2_checkpoint.pth.tar")
net.load_state_dict(dict["state_dict"])
end = time.time()
print(end - start)
print('///////')

crop_size = (512, 384)

while (i):
    ret, frame2 = cap.read()

    next = frame2

    # start = time.time()

    pim1 = cv2.resize(prvs, crop_size, interpolation=cv2.INTER_AREA)
    pim2 = cv2.resize(next, crop_size, interpolation=cv2.INTER_AREA)



    images = [pim1, pim2]


    # images = [prvs,next]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    start = time.time()
    result = net(im).squeeze()
    end = time.time()
    print(end - start)
    data = result.data.cpu().numpy().transpose(1, 2, 0)
    img = flow2img(data)
    # 写入之前最好设置一个清空文件夹操作
    cv2.imwrite(save_path + str(i) + '.png', img)    #如果想保存光流图片，以备后面调用flames2video得到光流video.


    cv2.imshow("window",img)
    kk = cv2.waitKey(20) & 0xff  #实时可视化光流图片，（人自己写好了flow2img函数）
    # Press 'e' to exit the video
    if kk == ord('e'):
        break
    #    plt.imshow(img)
    #    plt.show()
    i = i + 1
    prvs = next
# pic2video(path,crop_size)
# pic2video(path,crop_size)
