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
estimated_pim2pathN = "D:\Study\Datasets\pairs\\video64\\estimated154.png"
estimated_pim2pathP = "D:\Study\Datasets\pairs\\video64\\estimated150.png"

RGBflowpath = 'D:\Study\Datasets\pairs\\video64\Pair150n154.png'

#
# pim1path = "D:\Study\Datasets\pairs\\video64\\jinjing1.png"
# pim2path = "D:\Study\Datasets\pairs\\video64\\jinjing2.png"
# estimated_pim2path = "D:\Study\Datasets\pairs\\video64\\estimatedjinjing2prime.png"
# RGBflowpath = "D:\Study\Datasets\pairs\\video64\\Pairjinjing1n2.png"





# def run_pair(pim1path,pim2math,estimated_pim2path,RGBflowpath,method):













if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    # crop_size = (768, 512)
    crop_size = (512, 384)

    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("checkpoints/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])





    # load the image pair, you can find this operation in dataset.py
    # pim1 = read_gen(pim1path)
    # pim2 = read_gen(pim2path)
    pim1 = cv2.imread(pim1path)
    pim2 = cv2.imread(pim2path)
    # cv2.imwrite(pim1path, pim1)
    # cv2.imwrite(pim2path, pim2)

    pim1 = cv2.resize(pim1, crop_size, interpolation=cv2.INTER_LINEAR)
    pim2 = cv2.resize(pim2, crop_size, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(pim1path, pim1)
    cv2.imwrite(pim2path, pim2)

    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()


    # # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    # def writeFlow(name, flow):
    #     f = open(name, 'wb')
    #     f.write('PIEH'.encode('utf-8'))
    #     np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    #     flow = flow.astype(np.float32)
    #     flow.tofile(f)
    #     f.flush()
    #     f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)

    estimated_pim2N = warp_flow(pim1,data)  # 这个warp是把1warp到2
    estimated_pim2P = image_warp(pim2, data, mode='nearest') #  这个warp是把2warp回1
    cv2.imwrite(estimated_pim2pathN, estimated_pim2N)
    cv2.imwrite(estimated_pim2pathP, estimated_pim2P)


    img = flow2img(data)   #  data是flow array   这里直接得到bgr图片

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 下面四行 JINJING  改为用HV eoncde 光流
    img[:, :, 2] = img[:, :, 1]
    img[:, :, 1] = 255   #  此时的img为HSV的


    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # 写入之前最好设置一个清空文件夹操作
    cv2.imwrite(RGBflowpath, img)    #如果想保存光流图片，以备后面调用flames2video得到光流video.


