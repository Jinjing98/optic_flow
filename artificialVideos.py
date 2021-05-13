# -*- coding: UTF-8 -*-
import os
import cv2
import time

import cv2
import numpy as np

def pic2video(path, size):
    # path = 'output'#文件路径
    # filelist = len(os.listdir(path))  # 获取该目录下的所有文件名
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 25
    # file_path = path + 'out' + str(int(time.time())) + ".avi"  # 导出路径  mp4?
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # （'I','4','2','0' 对应avi格式）
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') #mp4

    video = cv2.VideoWriter(path,fourcc, fps, size)  # it want unsiged chars



    NonSigW =np.full([480,854,3], 255, dtype=np.uint8)
    NonSigB =np.full([480,854,3], 25, dtype=np.uint8)
    NonSigWcp = NonSigW.copy()
    NonSigBcp = NonSigB.copy()

    SigWinB = cv2.rectangle(NonSigBcp,(400,200),(450,280),(255,255,255),-1)
    SigBinW = cv2.rectangle(NonSigWcp,(400,200),(450,280),(25,25,25),-1)  # 50*80

    # cv2.imshow("", NonSigW)
    # cv2.waitKey(0)

    for i in range(10):
        for i in range(5):
            video.write(NonSigW)
        for i in range(20):
            video.write(SigBinW)



    video.release()  # 释放


if __name__ == '__main__':
    # pic2video('../Datasets/hand_test/hand_test_result/', (512, 384))
    # pic2video('D:\Study\Datasets\\video64_noise\\',(854,480))
    # save_path = '../Datasets/surgery_test/noisy_test_result/'
    # pic2video('../Datasets/surgery_test/noisy_test_result/',(512, 384))
    # pic2video('D:\Study\\Datasets\\moreCamBest\\NNflow\\',(512,384))
    #pic2video('D:\Study\\Datasets\\moreCamBestNontrembling\\set\\',(124,94))
    # pic2video('D:\Study\\Datasets\\moreCamBest\\GFflow\\',(124,94))
    # pic2video('D:\Study\\Datasets\\moreCamBestNontrembling\\set\\', (124, 94))
    pic2video('D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\\arti_videos\\BinW25_1hz.avi', (854,480))


    # img_1 = np.zeros([512, 512, 1], np.uint8)
    # # img_1.fill(255)
    # # or img[:] = 255
    # cv2.imshow('Single Channel Window', img_1)
    # print("image shape: ", img_1.shape)
    #
    # img_3 = np.zeros([512, 512, 3], dtype=np.uint8)
    # img_3.fill(255)
    # # or img[:] = 255
    # cv2.imshow('3 Channel Window', img_3)
    # print("image shape: ", img_3.shape)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # # Create a black image
    # img = np.zeros([480,854,1], np.float32)
    #
    # cv2.imshow("",img)
    # cv2.waitKey(0)

    # Draw a diagonal blue line with thickness of 5 px
    # img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    # img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)


