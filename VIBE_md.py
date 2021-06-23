import numpy as np
import os
import cv2


# https://github.com/yangshiyu89/VIBE/blob/master/vibe_test.py
#https://blog.csdn.net/qinglongzhan/article/details/82797413

def initial_background(I_gray, N):
    I_pad = np.pad(I_gray, 1, 'symmetric')
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    samples = np.zeros((height, width, N))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for n in range(N):
                x, y = 0, 0
                while (x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj]
    samples = samples[1:height - 1, 1:width - 1]
    return samples


def vibe_detection(I_gray, samples, _min, N, R):
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    segMap = np.zeros((height, width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            count, index, dist = 0, 0, 0
            while count < _min and index < N:
                dist = np.abs(I_gray[i, j] - samples[i, j, index])
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                r = np.random.randint(0, N - 1)
                if r == 0:
                    r = np.random.randint(0, N - 1)
                    samples[i, j, r] = I_gray[i, j]
                r = np.random.randint(0, N - 1)
                if r == 0:
                    x, y = 0, 0
                    while (x == 0 and y == 0):
                        x = np.random.randint(-1, 1)
                        y = np.random.randint(-1, 1)
                    r = np.random.randint(0, N - 1)
                    ri = i + x
                    rj = j + y
                    try:
                        samples[ri, rj, r] = I_gray[i, j]
                    except:
                        pass
            else:
                segMap[i, j] = 255
    return segMap, samples


videoname = 2
format = ".avi"
prefix = "D:\Study\Datasets\AAAtest\\"

def vibe_mask(videoname,format,prefix):
    prefix4mask = prefix + str(videoname) + "\\other\\"
    videopath = prefix + str(videoname) + format
    cap = cv2.VideoCapture(videopath)
    ret, frame = cap.read()

    N = 20
    R = 20
    _min = 2
    phai = 16
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    samples = initial_background(frame, N)
    i = 0
    while True:
        # path = os.path.join(rootDir, lists)
        # frame = cv2.imread(path)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, gray = cap.read()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        segMap, samples = vibe_detection(gray, samples, _min, N, R)
        cv2.imshow('segMap', segMap)
        kk = cv2.waitKey(20) & 0xff
        if kk == ord('q'):
            break
        # Press 's' to save the video
        elif kk == ord('s'):
            path_map = prefix4mask + "vibe_ID" + str(i) + ".png"
            print(path_map)
            cv2.imwrite(path_map, segMap)
        i += 1

    cv2.destroyAllWindows()
vibe_mask(videoname,format,prefix)

