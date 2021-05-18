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


def FN(videopath, sigDIR, String0, range, gridnumX, gridnumY):
        crop_size = (324, 274)
        size = (854, 480)
        PATHMAG = sigDIR + "FN" + "\size" + String0 + "\\" + "MAG" + "\\" + "SIGS_" + "FN"  + "_" + String0 + "_" + str(
            range[0]) + "_" + str(range[1]) + "mag.npy"
        PATHANG = sigDIR + "FN" + "\size" + String0 + "\\" + "ANG" + "\\" + "SIGS_" + "FN"  + "_" + String0 + "_" + str(
            range[0]) + "_" + str(range[1]) + "ang.npy"

        cap = cv2.VideoCapture(videopath)
        _, frame1 = cap.read()
        ret, frame1 = cap.read()

        prvs = frame1




        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        lower = fps * range[0]
        upper = fps * range[1]
        framenum = upper - lower



        gridWidth = int(width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
        gridHeight = int(height / gridnumY)  # 240

        SIGS_ang = np.zeros((framenum, gridnumY, gridnumX))
        SIGS_mag = np.zeros((framenum, gridnumY, gridnumX))






        start = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
        parser.add_argument("--rgb_max", type=float, default=255.)
        args = parser.parse_args()
        # initial a Net

        # net = FlowNet2C(args).cuda()
        # net = FlowNet2S(args).cuda()
        # net = FlowNet2CS(args).cuda()
        net = FlowNet2(args).cuda()
        # net = FlowNet2(args).cuda()
        # load the state_dict
        dict = torch.load("checkpoints/FlowNet2_checkpoint.pth.tar")  # flownet2
        net.load_state_dict(dict["state_dict"])
        end = time.time()
        print(end - start)
        print('///////')




        timepoint = 0
        i = 1
        # Till you scan the video
        start = time.time()

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

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # JINJING  改为用HV eoncde 光流
            img[:, :, 2] = img[:, :, 1]
            img[:, :, 1] = 255

            next_ang = img[..., 0].reshape(gridnumY, gridHeight,gridnumX, gridWidth)
            next_mag = img[..., 2].reshape(gridnumY, gridHeight, gridnumX, gridWidth)
            SIGS_ang[timepoint] = next_ang.mean(axis=(1, 3))
            SIGS_mag[timepoint] = next_mag.mean(axis=(1, 3))

            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            double_representation = cv2.addWeighted(img, 0.5, pim2, 0.5, 0)
            cv2.imshow("window", double_representation)

            cv2.imshow("window", img)
            kk = cv2.waitKey(20) & 0xff  # 实时可视化光流图片，（人自己写好了flow2img函数）
            # Press 'e' to exit the video
            if kk == ord('e'):
                break
            #    plt.imshow(img)
            #    plt.show()
            i = i + 1
            timepoint = timepoint + 1
            prvs = next
        # pic2video(save_path,crop_size)

        print(np.array(SIGS_ang).size)
        print(np.array(SIGS_mag).size)
        np.save(PATHANG, SIGS_ang)
        np.save(PATHMAG, SIGS_mag)
















