# Importing libraries
import cv2
import math
import numpy as np
from frames2video import pic2video
import time
from warp import *
from pyoptflow import HornSchunck, getimgfiles
import torch
import numpy as np
import argparse
from utils.flow_utils import flow2img
import cv2
from warp import *

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module





pim1path = "D:\Study\Datasets\pairs\\video64\\150.png"
pim2path = "D:\Study\Datasets\pairs\\video64\\154.png"
estimated_pim2pathN = "D:\Study\Datasets\pairs\\video64\\estimated154HS.png"
estimated_pim2pathP = "D:\Study\Datasets\pairs\\video64\\estimated150HS.png"
diff_pim2pathN = "D:\Study\Datasets\pairs\\video64\\diff154HS.png"
diff_pim2pathP = "D:\Study\Datasets\pairs\\video64\\diff150HS.png"
flow_path = 'D:\Study\Datasets\pairs\\video64\Pair150n154HS.png'


frame1 = cv2.imread(pim1path)
frame2 = cv2.imread(pim2path)
# crop_size = (512,384)
hsv_mask = np.zeros_like(frame1)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
i = 1
while(i):
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



    start = time.time()
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale = 0.5, levels = 7,
                                        winsize = 7, iterations = 5, poly_n = 9, poly_sigma = 1.1,flags = 0)
    end = time.time()
    print('GFcomputing time : ' + str(end-start))
    start = time.time()





# the 3 lines is exclusive for HS  to get a new different flow
    U, V = HornSchunck(prvs, next, alpha=20, Niter=15)
    end = time.time()
    print('HScomputing time : ' + str(end-start))
    flow = np.array([U.astype('float32'),V.astype('float32')])
    flow = np.transpose(flow, (1, 2, 0))



 # THESE LINES ARE FOR NN flow
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()
    start = time.time()

    net = FlowNet2(args).cuda()
    dict = torch.load("checkpoints/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    end = time.time()
    print('NNloading time : ' + str(end - start))
    images = [frame1, frame2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    start = time.time()
    flow = net(im)
    end = time.time()
    flow = flow.squeeze()
    print('NNcomputing time : ' + str(end-start))
    flow = flow.data.cpu().numpy().transpose(1, 2, 0)




















    start = time.time()
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees= False)   #in radians not degrees
    hsv_mask[..., 0] = ang * 180 / np.pi/2   #H : 0-180
    hsv_mask[..., 1] = 255
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX )   #nolinear transformation of value   sigmoid?
    end = time.time()
    print('vis time : ' + str(end-start))
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)


    # cv2.imshow('result_window', double_representation)
    # cv2.imshow('result_window', rgb_representation)
    # cv2.imwrite(flow_path,rgb_representation)



    estimated_pim2N = warp_flow(frame1,flow)  # 这个warp是把1warp到2
    estimated_pim2P = image_warp(frame2, flow, mode='nearest') #  这个warp是把2warp回1
    # cv2.imwrite(estimated_pim2pathN, estimated_pim2N)
    # cv2.imwrite(estimated_pim2pathP, estimated_pim2P)



    estimated_pim2P = cv2.cvtColor(estimated_pim2P, cv2.COLOR_BGR2GRAY)
    estimated_pim2N = cv2.cvtColor(estimated_pim2N, cv2.COLOR_BGR2GRAY)
    diff_P = estimated_pim2P - prvs
    # diff_P = -estimated_pim2P + prvs
    diff_N = -estimated_pim2N + next
    # cv2.imwrite(diff_pim2pathP, diff_P)
    # cv2.imwrite(diff_pim2pathN, diff_N)





    kk = cv2.waitKey(20) & 0xff
	# Press 'e' to exit the video
    if kk == ord('e'):
        break
	# Press 's' to save the video
    elif kk == ord('s'):
        cv2.imwrite('Optical_image.png', frame2)
        cv2.imwrite('HSV_converted_image.png', rgb_representation)
    i = 0
cv2.destroyAllWindows()

