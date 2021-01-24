# -*- coding: utf-8 -*-
import numpy as np
import cv2
from frames2video import pic2video
import warnings
import torch
import numpy as np
import argparse
import time
from models import *  # the path is depended on where you create this module

from utils.flow_utils import flow2img
import matplotlib.pyplot as plt


flow_path = "D:\Study\Datasets\\Cam\\NNflow\\"
flow_path = "D:\Study\\Datasets\\moreCamBest\\NNflow\\"



# crop_size = (768, 512)
crop_size = (512, 384)

# cap = cv2.VideoCapture("D:\Study\Datasets\\video64.mp4")
# cap = cv2.VideoCapture("D:\Study\Datasets\\noisy64.mp4")
cap = cv2.VideoCapture("D:\Study\Datasets\\test3L.mp4")
# cap = cv2.VideoCapture("D:\Study\Datasets\\test1L.mp4")
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("D:\Study\\Datasets\\moreCamBest.mp4")






# 获取第一帧
ret, frame1 = cap.read()


prvs = frame1
# prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)





# save_path = '../Datasets/surgery_test/surgery_test_result/'



# obtain the necessary args for construct the flownet framework
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
args = parser.parse_args()
# initial a Net

# net = FlowNet2C(args).cuda()
#net = FlowNet2S(args).cuda()
# net = FlowNet2CS(args).cuda()
net = FlowNet2(args).cuda()
# net = FlowNet2(args).cuda()
# load the state_dict
dict = torch.load("checkpoints/FlowNet2_checkpoint.pth.tar") #flownet2
net.load_state_dict(dict["state_dict"])
end = time.time()
print(end - start)
print('///////')


i = 1  # 控制实现的张数
SIGS_ang = []
SIGS_mag = []
while (i):
    ret, frame2 = cap.read()
    #
    if ret != True:
        break


    next = frame2
    # next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    # next = cv2.GaussianBlur(next, (21, 21), 0)

    # start = time.time()

    pim1 = cv2.resize(prvs, crop_size, interpolation=cv2.INTER_LINEAR)
    pim2 = cv2.resize(next, crop_size, interpolation=cv2.INTER_LINEAR)


    images = [pim1, pim2]



    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
    # print (cap.get(3))
    start = time.time()
    result = net(im).squeeze()
    # result = net(im)
    end = time.time()
    print(end - start)
    data = result.data.cpu().numpy().transpose(1, 2, 0)
    img = flow2img(data)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #  JINJING  改为用HV eoncde 光流
    img[:,:,2] = img [:,:,1]
    img[:,:,1] = 255

    SIG_ang = np.mean(img[..., 0])
    SIG_mag = np.mean(img[..., 2])
    SIGS_ang.append(SIG_ang)
    SIGS_mag.append(SIG_mag)



    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)










    # 写入之前最好设置一个清空文件夹操作
    # cv2.imwrite(save_path + str(i) + '.png', img)    #如果想保存光流图片，以备后面调用flames2video得到光流video.

    # rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    double_representation = cv2.addWeighted(img,0.5,pim2, 0.5, 0)
    # cv2.imshow("window",double_representation)
    # cv2.imwrite(flow_path + str(i) + '.png', img)
    cv2.imshow("window", img)
    kk = cv2.waitKey(20) & 0xff  #实时可视化光流图片，（人自己写好了flow2img函数）
    # Press 'e' to exit the video
    if kk == ord('e'):
        break
    #    plt.imshow(img)
    #    plt.show()
    i = i + 1
    prvs = next
# pic2video(save_path,crop_size)

print(np.array(SIGS_ang).size)
print(np.array(SIGS_mag).size)
# np.save(flow_path+"SIGS_ang.txt",SIGS_ang)
# np.save(flow_path+"SIGS_mag.txt",SIGS_mag)
