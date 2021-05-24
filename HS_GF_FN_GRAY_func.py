import cv2
import math
import time
import  numpy as np
from pyoptflow import HornSchunck, getimgfiles
from frames2video import pic2video
#+"\\"+note


import cv2
#from frames2video import pic2video
import warnings
import torch
import numpy as np
import argparse
import time
from models import *  # the path is depended on where you create this module

from utils.flow_utils import flow2img

# we require all 512*384 video if we want to use FN
def FN(videopath, sigDIR, String0, ranges, gridnumX, gridnumY):
        crop_size =  (gridnumX,gridnumY)
        # gridnumX = 512
        # gridnumY = 384
        PATHMAG = sigDIR + "FN" + "\\"+"mag"+"\size" + String0 + "\\" + "SIGS_" + "FN" + "_" + String0 + "_" + str(
            ranges[0]) + "_" + str(ranges[1]) + "mag.npy"
        PATHANG = sigDIR + "FN" +"\\"+"ang"+ "\size" + String0 + "\\" + "SIGS_" + "FN"+ "_" + String0 + "_" + str(
            ranges[0]) + "_" + str(ranges[1]) + "ang.npy"

        cap = cv2.VideoCapture(videopath)
        _, frame1 = cap.read()
        ret, frame1 = cap.read()

        prvs = frame1




        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        lower = fps * ranges[0]
        upper = fps * ranges[1]
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

            if i < lower:  # 直接从好帧开始运行
                i += 1
                continue
            if i >= upper:
                break

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

            next_ang = (img[..., 0]*2).reshape(gridnumY, 1,gridnumX, 1)#gridheight grid weight 1 1
            next_mag = img[..., 2].reshape(gridnumY, 1, gridnumX, 1)
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
            if timepoint == 148:
                print("")
            prvs = next
        # pic2video(save_path,crop_size)

        print(np.array(SIGS_ang).size)
        print(np.array(SIGS_mag).size)
        np.save(PATHANG, SIGS_ang)
        np.save(PATHMAG, SIGS_mag)








def HS(videopath,sigDIR,String0,ranges,gridnumX,gridnumY):# videopath,sigDIR,mode,String0,ranges,gridnumX,gridnumY

	PATHMAG = sigDIR + "HS" + "\\"+"mag"+ "\size" + String0 +"\\" + "SIGS_" + "HS" + "_" + String0 + "_" + str(
		ranges[0]) + "_" + str(ranges[1]) + "mag.npy"
	PATHANG = sigDIR +"HS" + "\\"+"ang"+ "\size" + String0 + "\\" + "SIGS_" + "HS" + "_" + String0 + "_" + str(
		ranges[0]) + "_" + str(ranges[1]) + "ang.npy"
	#mode+"\size"+String0+"\\"+note+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(tim

	size = (854, 480)

	cap = cv2.VideoCapture(videopath)
	_, frame1 = cap.read()
	# Convert to gray scale
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	lower =  fps*ranges[0]
	upper =  fps*ranges[1]
	framenum = upper - lower

	hsv_mask = np.zeros_like(frame1)
	hsv_mask[..., 1] = 255

	gridWidth = int(width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
	gridHeight = int(height / gridnumY)  # 240

	SIGS_ang = np.zeros((framenum, gridnumY, gridnumX))
	SIGS_mag = np.zeros((framenum, gridnumY, gridnumX))

	timepoint = 0
	i = 1
	# Till you scan the video
	start = time.time()
	while (i):

		# Capture another frame and convert to gray scale
		ret, frame2 = cap.read()

		if i < lower:  #   直接从好帧开始运行
		    i += 1
		    continue
		if i >=upper:
			break

		if ret != True:
			break
		next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



		U, V = HornSchunck(prvs, next, alpha=10, Niter=10)


		mag, ang = cv2.cartToPolar(U, V, angleInDegrees=False)  # in radians not degrees

		# Set image hue according to the angle of optical flow   HUE  1st d of hsv
		hsv_mask[..., 0] = ang * 180 / np.pi #/ 2 jinjing
		# Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
		hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255,
										 cv2.NORM_MINMAX)  # nolinear transformation of value   sigmoid?

		prvs = next
		# next = next.reshape(gridnumY, gridHeight, gridnumX, gridWidth)
		next_ang = hsv_mask[..., 0].reshape(gridnumY, 1, gridnumX, 1)
		next_mag = hsv_mask[..., 2].reshape(gridnumY, 1, gridnumX, 1)
		SIGS_ang[timepoint] = next_ang.mean(axis=(1, 3))
		SIGS_mag[timepoint] = next_mag.mean(axis=(1, 3))

		rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
		double_representation = cv2.addWeighted(rgb_representation, 3, frame2, 0.5, 0)

		cv2.imshow('result_window', double_representation)
		# # cv2.imshow('result_window', rgb_representation)
		# cv2.imwrite(flow_path + str(i) + '.png', rgb_representation)

		kk = cv2.waitKey(20) & 0xff
		# Press 'e' to exit the video
		if kk == ord('e'):
			break
		# prvs = next
		i += 1
		timepoint += 1
	end = time.time()
	print("flow computing time: " + str(end - start))

	cap.release()
	cv2.destroyAllWindows()

	np.save(PATHANG,SIGS_ang)
	np.save(PATHMAG,SIGS_mag)


def GF(videopath,sigDIR,String0,ranges,gridnumX,gridnumY):
	size = (854, 480)
	PATHMAG = sigDIR + "GF" +"\\"+"mag"+ "\size" + String0 + "\\"+ "SIGS_" + "GF" + "_" + String0 + "_" + str(
		ranges[0]) + "_" + str(ranges[1]) + "mag.npy"
	PATHANG = sigDIR + "GF" +"\\"+"ang"+ "\size" + String0 +  "\\" + "SIGS_" + "GF"  + "_" + String0 + "_" + str(
		ranges[0]) + "_" + str(ranges[1]) + "ang.npy"

	cap = cv2.VideoCapture(videopath)
	_, frame1 = cap.read()
	# Convert to gray scale
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	lower = fps * ranges[0]
	upper = fps * ranges[1]
	framenum = upper - lower

	hsv_mask = np.zeros_like(frame1)
	hsv_mask[..., 1] = 255

	gridWidth = int(
		width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
	gridHeight = int(height / gridnumY)  # 240

	SIGS_ang = np.zeros((framenum, gridnumY, gridnumX))
	SIGS_mag = np.zeros((framenum, gridnumY, gridnumX))

	timepoint = 0
	i = 1
	# Till you scan the video
	start = time.time()
	while (i):

		# Capture another frame and convert to gray scale
		ret, frame2 = cap.read()
		if i == 349:
			print()

		if i < lower:  # 直接从好帧开始运行
			i += 1
			continue
		if i >= upper:
			break

		if ret != True:
			break
		next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



		flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=5, iterations=5,
											poly_n=5,
											poly_sigma=1.1, flags=0)

		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)




	# Set image hue according to the angle of optical flow   HUE  1st d of hsv
		hsv_mask[..., 0] = ang * 180 / np.pi #/2 jinjing
		# Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
		hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255,
										 cv2.NORM_MINMAX)  # nolinear transformation of value   sigmoid?

		prvs = next
		# next = next.reshape(gridnumY, gridHeight, gridnumX, gridWidth)
		next_ang = hsv_mask[..., 0].reshape(gridnumY, 1, gridnumX, 1)
		next_mag = hsv_mask[..., 2].reshape(gridnumY, 1, gridnumX, 1)
		SIGS_ang[timepoint] = next_ang.mean(axis=(1, 3))
		SIGS_mag[timepoint] = next_mag.mean(axis=(1, 3))

		rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
		double_representation = cv2.addWeighted(rgb_representation, 3, frame2, 0.5, 0)

		cv2.imshow('result_window', double_representation)
		# # cv2.imshow('result_window', rgb_representation)
		# cv2.imwrite(flow_path + str(i) + '.png', rgb_representation)

		kk = cv2.waitKey(20) & 0xff
		# Press 'e' to exit the video
		if kk == ord('e'):
			break

		i += 1
		timepoint += 1
	end = time.time()
	print("flow computing time: " + str(end - start))

	cap.release()
	cv2.destroyAllWindows()

	np.save(PATHANG, SIGS_ang)
	np.save(PATHMAG, SIGS_mag)

def GRAY(videopath,sigDIR,String0,time_range,gridnumX,gridnumY ):  # choose ranges given a video length to collect npy
    #meansigarrayPath = meansigarrayDIR+mode+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"
    sigpath = sigDIR+"gray"+"\size"+String0+"\\"+"SIGS_"+"gray"+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"


    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    _, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    lower = fps * time_range[0]
    upper = fps * time_range[1]
    framenum = upper - lower


    gridWidth = int(width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
    gridHeight = int(height / gridnumY)  # 240
    SIGS_gray = np.zeros((framenum, gridnumY, gridnumX))  # 4,3


    i = 1
    timepoint = 0
    start1 = time.time()
    while (i):

        ret, frame2 = cap.read()
        if i < lower:  # 直接从好帧开始运行
            i += 1
            continue
        if i >= upper:
            break
        if ret != True:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


        start = time.time()
        # before it took around 3 mins, now it is way faster based on numpy vectorization
        # https://codingdict.com/questions/179693
        next = next.reshape(gridnumY, 1, gridnumX, 1)

        # print(next[0,:,0,:])

        SIGS_gray[timepoint] = next.mean(axis=(1, 3))

        end = time.time()
        print('time to compute mean signal for windows' + str(end - start))
        timepoint += 1
        i += 1
    end1 = time.time()
    print("total time to collect mean gray and mean flow: " + str(end1 - start1))

    # np.save(sigpath+"GFSIGS_ang.txt",SIGS_ang)
    # np.save(sigpath+"GFSIGS_mag.txt",SIGS_mag)
    np.save(sigpath, SIGS_gray)
    # np.save("D:\\Study\\Datasets\\AEXTENSION\\Cho80_extension\\static_cam\\pulseNstatic\\1\\gray\\size854_480\\SIGS_gray854_480.npy",SIGS_gray)


