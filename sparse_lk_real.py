import numpy as np
import cv2

cap = cv2.VideoCapture("D:\Study\Datasets\\video64.mp4")
# cap = cv2.VideoCapture("D:\Study\Datasets\\noisy64.mp4")
cap = cv2.VideoCapture("D:\Study\Datasets\\morecam.mp4")
# cap = cv2.VideoCapture('D:\Study\Datasets\signalVideos\staticCam\\pptcardNresp_turbG2.avi')


# cap = cv2.VideoCapture(0)

# ShiTomas角点检测的参数
feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=3, blockSize=7)

# 金字塔LK算法参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))
# 创建随机颜色
color = np.random.randint(0, 255, (1000, 3))

while(1):
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('',old_gray)
    #
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break



    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)#compute feature points just for one time
    init_size = np.size(p0)
    mask = np.zeros_like(old_frame)
    j = 1

    while (1):
        ret, frame = cap.read()

        # if ret is True:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # else:
        #     break

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        #jinjing
        if (p1 is None) :  # 跟丢之后重修找特征点
            break
        good_new = p1[st == 1]
        #good_old = p0[st == 1]

        if (np.size(good_new)<np.maximum(0.7 * init_size,15)):   #特征点太少 存在跟丢可能时也重新找
            break

        print(np.size(good_new))
        background = np.zeros_like(frame)   #jinjing

        for i, new in enumerate(good_new):
            a, b = new.ravel()
           # c, d = old.ravel()
            #mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

            frame2 = cv2.circle(background, (a, b), 5, color[1].tolist(), -1)
        img = cv2.add(frame2, frame)

        cv2.imshow('frame', img)
        j += 1
        cv2.imwrite("D:\Study\Datasets\signalVideos\\test\\"+str(j)+".png",img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()