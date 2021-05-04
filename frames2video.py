# -*- coding: UTF-8 -*-
import os
import cv2
import time

size = (512, 384)
def pic2video(path, size):
    # path = 'output'#文件路径
    filelist = len(os.listdir(path))  # 获取该目录下的所有文件名
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 25
    file_path = path + 'out' + str(int(time.time())) + ".avi"  # 导出路径
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # （'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(file_path, fourcc, fps, size)

    for i in range(filelist):
        i = path + '/' + str(i) + '.png'
        img = cv2.imread(i)
        video.write(img)
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
    pic2video('D:\Study\Datasets\signalVideos\\test\\', (720,288))



