import cv2
import numpy as np


def gettruth(videopath):
    cap = cv2.VideoCapture(videopath)

    _, frame1 = cap.read()



    i = 1

    while (i):

        ret, frame2 = cap.read()

        if ret != True:
            break



        cv2.imshow("",frame2)

        kk = cv2.waitKey(0) & 0xff  # 实时可视化光流图片，（人自己写好了flow2img函数）
        # Press 'e' to exit the video
        if kk == ord('s'):
            cv2.imwrite(videopath[:-4]+"_"+str(i)+".png",frame2)


        cv2.waitKey(0)

        i += 1
gettruth("D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\14.avi")