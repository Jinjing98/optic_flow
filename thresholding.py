import cv2
# img = cv2.imread('C:\Users\xjj89\Desktop\\7_12_5_20_1_0.11_1.2magmap.png')
img = cv2.imread("D:\Study\Datasets\AATEST\\new_short\8\gray\size512_384\\jinjing.png",0)
img2 = cv2.imread("D:\Study\Datasets\AATEST\\new_short\8\gray\size512_384\\jinjing.png")

# img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
# # dst = cv2.Mat()
# # // You can try more different parameters
# cv2.threshold(src, dst, 177, 200, cv.THRESH_BINARY);
# # cv.imshow('canvasOutput', dst);
# # src.delete();
# # dst.delete();
# cv2.imshow("",img)
# cv2.waitKey(0)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# img = cv.imread('D:\Study\Datasets\AATEST\\new_short\8\gray\size512_384\\jinjing2.png',0)
# global thresholding
ret1,th1 = cv.threshold(img,33,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = [str(ret1)+' Original Noisy Image','Histogram','Global Thresholding',
          str(ret2)+' Original Noisy Image','Histogram',"Otsu's Thresholding",
          str(ret3)+' Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()