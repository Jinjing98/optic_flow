import numpy as np
import pylab as pl
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import time
import pylab as plt
from scipy import stats
from HS_GF_FN_GRAY_func import HS,GF,GRAY,GRAY_free,FN
from test import draw2pointsRAW_FFT,on_EVENT_LBUTTONDOWN
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from evaluation import *
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import norm
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

import numpy.ma as ma

def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    # acorr = result[:] / (x.var() * np.arange(n,0, -1))

    # lag = np.abs(acorr).argmax() + 1
    # r = acorr[lag-1]
    # if np.abs(r) > 0.5:
    #   print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else:
    #   print('Appears to be not autocorrelated')
    return acorr

#
# train2 = autocorr(train1)
# x_axix2 = [i for i in
#            range(len(train2))]  # [i for i in range(100)] if len(train1) == 100 else [i for i in range(len(train1))]

# peaks, _ = find_peaks(train2, height=0)
# plt.plot(x_axix2, train2)
# plt.xlabel('frame ID')
# plt.plot(peaks, train2[peaks], "x")
# plt.plot(np.zeros_like(train2), "--", color="gray")
# plt.title("ACF with peak " + str(Y) + "_" + str(X))
# plt.show()


def playvideowithmask(fps,ranges,videopath, mask,maskNO,outimgpath,outimgpathNO):
    cap = cv2.VideoCapture(videopath)
    lower = fps * ranges[0]
    upper = fps * ranges[1]
    _, frame1 = cap.read()
    i = 1

    # Till you scan the video
    while (i):

        # Capture another frame and convert to gray scale
        ret, frame = cap.read()
        if i < lower:  # 直接从好帧开始运行
            i += 1
            continue
        if i >= upper:
            break
        if ret != True:
            break
    # while 1:
    #     ret, frame = cap.read()  # 如果想保存光流图片，以备后面调用flames2video得到光流video.
    #     if ret!=True:
    #         break

        # double_representation = mask+frame#cv2.addWeighted(frame, 1, 255-mask, 1, 0.2,dtype = cv2.CV_32F)
        frameYES = cv2.bitwise_and(frame, frame, mask=np.uint8(mask))
        frameNO = cv2.bitwise_and(frame, frame, mask=np.uint8(maskNO))
        i += 1


        # cv2.imshow("window", frameYES)
        cv2.imshow("window",frameNO)

        kk = cv2.waitKey(20) & 0xff  # 实时可视化光流图片，（人自己写好了flow2img函数）
        # Press 'e' to exit the video
        if kk == ord('e'):
            #cv2.imwrite(outimgpath,frameYES)
            # cv2.imwrite(outimgpathNO,frameNO)

            break


def peaktest(idx,fft_result):
    magnitude = np.abs(np.absolute(fft_result[idx]))
    mag_left = np.abs(np.absolute(fft_result[idx-1]))
    mag_right = np.abs(np.absolute(fft_result[idx+1]))
    return (magnitude>mag_left) and (magnitude>mag_right)
def significant_peaktest(idx,fft_result,top10_ID):
    magnitude = np.abs(np.absolute(fft_result[idx]))
    mean_mag = np.mean(np.abs(np.absolute(fft_result[top10_ID])))
    # mag_left = np.abs(np.absolute(fft_result[idx-1]))
    return (magnitude>mean_mag*1)#1.2)
# def check2Harm(secondHarmonicIDX,mask2D,bestIDX,topharmonic_ID,x,y):
#     if (int(secondHarmonicIDX) in topharmonic_ID[:, y, x]):
#         return True,secondHarmonicIDX
#     elif (int((secondHarmonicIDX + 1)) in topharmonic_ID[:, y, x]):
#         secondHarmonicIDX = secondHarmonicIDX + 1
#         return True,secondHarmonicIDX
#     elif (int((secondHarmonicIDX - 1))in topharmonic_ID[:, y, x]):
#         secondHarmonicIDX = secondHarmonicIDX - 1
#         return True,secondHarmonicIDX
#     else:
#         mask2D[y, x] = False
#         bestIDX[y, x] = 0
#         #three
#         maskset[2:, y, x] = False
#         return False,secondHarmonicIDX
#
#
# def check3Harm(thirdHarmonicIDX,mask2D,bestIDX,topharmonic_ID,x,y):
#     if (int(thirdHarmonicIDX) in topharmonic_ID[:, y, x]):
#         return True,thirdHarmonicIDX
#     elif (int((thirdHarmonicIDX + 1)) in topharmonic_ID[:, y, x]):
#         thirdHarmonicIDX = thirdHarmonicIDX + 1
#         return True,thirdHarmonicIDX
#     elif (int((thirdHarmonicIDX - 1))  in topharmonic_ID[:, y, x]):
#         thirdHarmonicIDX = thirdHarmonicIDX - 1
#         return True,thirdHarmonicIDX
#     elif (int((thirdHarmonicIDX + 2)) in topharmonic_ID[:, y, x]):
#         thirdHarmonicIDX = thirdHarmonicIDX + 2
#         return True,thirdHarmonicIDX
#     elif (int((thirdHarmonicIDX - 2)) in topharmonic_ID[:, y, x]):
#         thirdHarmonicIDX = thirdHarmonicIDX - 2
#         return True,thirdHarmonicIDX
#     else:
#         mask2D[y, x] = False
#         bestIDX[y, x] = 0
#         #three
#         maskset[2:, y, x] = False
#         return False,thirdHarmonicIDX
#


def AFC_test(peaks_int,estimated_frame):  #  peaks int is with length 3

    delta_frames = peaks_int-estimated_frame
    # np.min(np.absolute(delta_frames)) < 3
    print(delta_frames)
    return np.min(np.absolute(delta_frames)) < 3


def test1(array_3d,pos_Y_from_fft,precision4given,top10_ID,topharmonic_ID,df,given_freq,maskNfreqID_infoMat,num4indexY,num4indexX):  # try to reduce FN

    gridnumy = top10_ID.shape[1]
    gridnumx = top10_ID.shape[2]

    ID_expect = given_freq/df
    top10_ID_filter = np.where(top10_ID *df< given_freq+ precision4given , top10_ID, 0)#  to solve the issue of 1.3/2.6
    top10_ID_filter = np.where(top10_ID_filter *df> given_freq- precision4given , top10_ID_filter, 0)#  to solve the issue of 1.3/2.6
    mask2D =  np.any(top10_ID_filter,axis= 0)#.astype(np.float64)
    bestIDX = np.zeros([gridnumy,gridnumx])
    bestStr = np.zeros_like(mask2D).astype(complex)
    closest_layer_number = np.argsort(-np.abs(top10_ID_filter-ID_expect),axis=0)[-1]#just 4 non-mask region is useful!
# how to attach the mask effeciently?

    ##one
    maskset[:] = mask2D.copy()



    real_totalNbr = len(array_3d[:,0,0])
    len4corr = len(array_3d[real_totalNbr//2+1:])
    peaks_int_set = np.zeros([3,num4indexY,num4indexX]).astype(int)
    corr_set = np.zeros([len4corr, num4indexY, num4indexX])


    for y in range(gridnumy):
        for x in range(gridnumx):
            if mask2D[y,x]:
                if y == 65 and x==249:
                    print()
                index = top10_ID_filter[closest_layer_number[y,x],y,x]
                bestIDX[y, x] = index

                # secondHarmonicIDX = 2*bestIDX[y,x]
                # thirdHarmonicIDX = 3*bestIDX[y,x]
                # if significant_peaktest(index,pos_Y_from_fft[:,y,x], top10_ID[:,y,x]) == False:  # hope to help removing irreg
                #     mask2D[y, x] = False
                #     bestIDX[y, x] = 0
                #     #two
                #     maskset[1:,y,x] = False
                #     continue

                #
                # flag2,secondHarmonicIDX =  check2Harm(secondHarmonicIDX,mask2D,bestIDX,topharmonic_ID,x,y)
                # flag3,thirdHarmonicIDX = check3Harm(thirdHarmonicIDX,mask2D,bestIDX,topharmonic_ID,x,y)




                #
                #
                #
                # freq_frame = np.rint(
                #     np.where(maskNfreqID_infoMat[1] == 0, 0, fps / (maskNfreqID_infoMat[1] * df))).astype(int)
                freq_frame = round(fps/(bestIDX[y,x]*df))
                data = array_3d[:,y,x]
                corr = autocorr(data)
                corr_set[:, y, x] = corr
                peaks_int, _ = find_peaks(corr, height=0,prominence=0,threshold=0.005)
                #The prominence of a peak measures how much a peak stands out from the surrounding baseline
                # of the signal and is defined as the vertical distance between the peak and its lowest contour line.
                if len(peaks_int) < 4:
                    peaks_int_set[:len(peaks_int), y, x] = peaks_int
                else:
                    peaks_int_set[:3, y, x] = top3_peaks(peaks_int, corr, num4indexY, num4indexX)
                #  acf test!
                if np.min(np.absolute(peaks_int_set[:,y,x]-freq_frame)) > 3:
                        print("acf:",peaks_int_set[:,y,x]-freq_frame)
                        mask2D[y, x] = False
                        bestIDX[y, x] = 0
                        #two
                        maskset[1,y,x] = False
                        continue






                bestID = int(bestIDX[y, x])
                bestStr[y, x] = pos_Y_from_fft[bestID, y, x]






    bestStr = np.abs(np.absolute(bestStr))  #  finally use  mask2D(bool) bestIDX bestStr
    print("max beststr,", np.max(bestStr))
    maskNfreqID_infoMat[0] = mask2D.astype(np.uint32)    #mask
    maskNfreqID_infoMat[1] = bestIDX#index

    return maskNfreqID_infoMat,bestStr   #  i can save up here!

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#normalize to 0-255
def test2(bestStr,maskNfreqID_infoMat,df): # require relative strong strength
    # bestStr1D = bestStr.flatten()
    # bestStr1D = -np.sort(-bestStr1D)
    maxStr = np.max(bestStr)
    # # bestStr1D = bestStr1D/maxStr
    #
    # absStr0 = 0
    # mask2D = np.where(bestStr>absStr0,True,False)
    # mask2D = mask2D.astype(np.uint32)


    # mask2D = maskNfreqID_infoMat[0].copy




    # peak_frames =
    # a = peaks_int_set
    #     else:
    #         mask2D[y, x] = False
    #         bestIDX[y, x] = 0
    #         #three
    #         maskset[2:, y, x] = False
    #         return False,secondHarmonicIDX






#
#
#
#
#     # maskNfreqID_infoMat[1] = np.where(mask2D == False, int(0),maskNfreqID_infoMat[1])
#
# #vis and save threshold0
#     img = plt.imshow(maskNfreqID_infoMat[1]*df, cmap='hot', interpolation='nearest')
#     plt.colorbar(img)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.title("frequency distribution of mask with threshold0")
#     plt.savefig(path_prefix + "_freqmap.png")
#     plt.show()
#
#
#     img2 = plt.imshow(bestStr, cmap='hot', interpolation='nearest')
#     plt.colorbar(img2)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.title("\"periodicness\" magnitude distribution of mask without threshold")
#     plt.savefig(path_prefix + "_magmap.png")
#     plt.show()
#     # normalize to 0-255 before ostu
#
#
#














    bestStr = (bestStr * 255 / maxStr).astype(np.uint8) if maxStr != 0 else bestStr.astype(np.uint8)
    # if gridnumx == 512:
    #     bestStr = cv2.GaussianBlur(bestStr, (5, 5), 0)## based on the raw num4x * num4y
    # elif gridnumx == 128:
    #     bestStr = cv2.GaussianBlur(bestStr, (5, 5), 0)
    # else:
    #     bestStr = cv2.GaussianBlur(bestStr, (5, 5), 0)
    bestStr = cv2.GaussianBlur(bestStr, (5, 5), 0)
    newthr, result_img = cv2.threshold(bestStr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # newmask = np.where(bestStr>newthr,True,False).astype(np.uint32)if maxStr!=0 else False
    # four
    maskset[2] = result_img
    # np.save(masksetpath,maskset)

    plt.subplot(1, 3, 1)
    plt.title("(a) nbr", y=-0.18)
    plt.imshow(maskset[0], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(path_prefix + "filtering.png")
    # plt.show()
    plt.subplot(1, 3, 2)
    plt.title("(b) acf",y = -0.18)
    plt.imshow(maskset[1], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(path_prefix + "_test1.png")
    # plt.show()
    plt.subplot(1, 3, 3)
    plt.title("(c)ostu",y = -0.18)
    plt.imshow(maskset[2], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    # # plt.savefig(path_prefix + "_test2.png")
    # # plt.show()
    # plt.subplot(2, 2, 4)
    # plt.title("(d)",y = -0.18)
    # plt.imshow(maskset[3], cmap="gray")
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(path_prefix + "_update.png")
    plt.show()
#
# # individually visulise final mask
#     plt.imshow(maskset[1], cmap="gray")
#     plt.xticks([])
#     plt.yticks([])
#     # plt.savefig(path_prefix + "_test3.png")
#     plt.show()







    return maskNfreqID_infoMat

def top3_peaks(peaks_int,corr,num4indexY,num4indexX):
    peaks_int_new = np.zeros_like(peaks_int)[:3]
    corr4peaks = corr[peaks_int]
    peaks_order_list = np.argsort(-np.abs(np.absolute(corr4peaks)),axis=0)
    peaks_ID = peaks_order_list[:3]
    print("inter_3ID:",peaks_ID)
    peaks_int_new = peaks_int[peaks_ID]
    print("peaks_int_new:",peaks_int_new)
    #
    # plt.plot(corr)
    # plt.xlabel('frame ID')
    # plt.plot(peaks_int, corr[peaks_int], "x")
    # plt.plot(np.zeros_like(corr), "--", color="gray")
    # plt.show()








    return peaks_int_new


def fft_window(conv_times,visuliseFFT,visuliseRAW,top,top4harmonic,fps,givenfreq,precision4given,path4signal,sample_freq,path_prefix ):  # 默认为half

    array_3d = np.load(path4signal)
    # array_3d = array_3d-np.average(array_3d,axis=0)
    #
    #
    # avg = np.average(array_3d,axis=0)
    # std = np.std(array_3d,axis=0)
    # std_mask = (std!=0)
    # array_3d = np.where(std_mask == False, array_3d-avg,(array_3d-avg)/std)

    # if conv_times:
    #     array_3d2 =(array_3d/array_3d.max())**conv_times
    #     array_3d = array_3d*array_3d2

    num4indexY =  array_3d.shape[1]
    num4indexX =  array_3d.shape[2]
    maskNfreqID_infoMat = np.zeros((2,num4indexY,num4indexX),np.uint32)  #  change 2 to 5  (+25,50,75)





    start = time.time()

    #speed up computation!

    step = round(fps / sample_freq)
    array_3d = array_3d[::step]
    real_totalNbr = array_3d.shape[0]
    real_sample_freq = float(fps) / step
    df = real_sample_freq / (real_totalNbr - 1)  # 0.03HZ
    pos_Y_from_fft = np.zeros([real_totalNbr//2,num4indexY,num4indexX],np.complex128)



    #how to speed here up! seems no way!no standard fft func optimized for numpy
    for i in range(num4indexY):
        for j in range(num4indexX):
            y = array_3d[:,i,j]

            Y = np.fft.fft(y) * 2 / real_totalNbr  # *2/N 反映了FFT变换的结果与实际信号幅值之间的关系
            pos_Y_from_fft[:,i,j] = Y[:Y.size // 2]  # 10  2

            # corr = autocorr(y)
            # corr_set[:,i,j] = corr
            # # x_axix2 = [i for i in [i for i in range(100)] if len(
            # #            range(len(train2))]  #train1) == 100 else [i for i in range(len(train1))]
            #
            # peaks_int, _ = find_peaks(corr, height=0)
            # if len(peaks_int) < 4:
            #     peaks_int_set[:len(peaks_int), i, j] = peaks_int
            # else:
            #     peaks_int_set[:3, i, j]= top3_peaks(peaks_int,corr,num4indexY,num4indexX)



            if real_totalNbr//2 < 6:
                print("the number of sampled frames is too less! Please increase <time_range> or <realfreq4samples> in the main!")
                return False, df

    order_list = np.argsort(-np.abs(np.absolute(pos_Y_from_fft)),axis=0)

    top10_ID = order_list[1:(top+1)]
    topharmonic_ID = order_list[1:(top4harmonic+1)]
    _,bestStr = test1(array_3d,pos_Y_from_fft,precision4given,top10_ID,topharmonic_ID, df, givenfreq, maskNfreqID_infoMat,num4indexY,num4indexX )


    test2(bestStr, maskNfreqID_infoMat, df)
    gridwidth = int(imgx / num4indexX)
    gridheight = int(imgy / num4indexY)

    # for interation  this is only utilized for the mask before level 3 of maskset
    # if we want to interact with other mask just change here!
    mask_2d = maskset[1]#maskNfreqID_infoMat[0]  #  i can change here according to what i want to see
    y_idx = np.nonzero(mask_2d)[0]
    x_idx = np.nonzero(mask_2d)[1]
    mask_2d = np.zeros((imgy, imgx))  # full resolution of final mask2D before amp test
    for y, x in zip(y_idx, x_idx):
        mask_2d[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask_2d = mask_2d.astype(np.float32) * 255
    mask_2dCP = mask_2d.copy()
    # from test import draw2pointsRAW_FFT, on_EVENT_LBUTTONDOWN
    cv2.namedWindow("image")
    img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP = (mask_2dCP,gridwidth, gridheight,pos_Y_from_fft,visuliseFFT,visuliseRAW,df,path4signal,array_3d,maskNfreqID_infoMat[1])
   # this func can interatively show curve
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP)
    cv2.imshow("image", mask_2dCP)
    cv2.waitKey(0)#  non vis interation!



## additinal seperataly save the final mask
    mask_path = path_prefix+"_mask.png"
    mask_2d = maskset[1]#maskNfreqID_infoMat[0]
    y_idx = np.nonzero(mask_2d)[0]
    x_idx = np.nonzero(mask_2d)[1]
    mask_2d = np.zeros((imgy, imgx))  # full resolution of final mask2D before amp test
    for y, x in zip(y_idx, x_idx):
        mask_2d[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask_2d = mask_2d.astype(np.float32) * 255
    cv2.imwrite(mask_path, mask_2d)#mask_2d


    end = time.time()
    print('fft time for all windows: ' + str(end - start))
   # np.save(infoMatPath,maskNfreqID_infoMat)#infomat now is maskNfreqID_infoMat
    return True,df,real_totalNbr


def pixelwisesig_generate(mode):

    if mode == "gray":
        GRAY(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )
    if mode == "GF":
        GF(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )
    if mode == "HS":
        HS(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )
    if mode == "FN":
        FN(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )



def gridsig_generate():
    path2sig = meansigarrayDIR + mode + "\\" + note + "\size" + str(imgx) + "_" + str(
        imgy) + "\\" +"SIGS_"+mode+"_"+str(time_range[0])+"_"+str(time_range[1])+note+".npy"
    newpath2sig = meansigarrayDIR + mode + "\\" + note + "\size" + str(gridnumx) + "_" + str(
        gridnumy) + "\\" + "SIGS_"+mode+"_"+str(time_range[0])+"_"+str(time_range[1])+note+".npy"

    try:
        sig = np.load(path2sig)
    except IOError:
        pixelwisesig_generate(mode)
        sig = np.load(path2sig)
    sig = sig.reshape(-1, gridnumy, int(imgy / gridnumy), gridnumx, int(imgx / gridnumx))
    newsig = sig.mean(axis=(2, 4))
    np.save(newpath2sig, newsig)











# extensionDir = "D:\Study\Datasets\AAAtest\\"#"D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulse_today\\"
extensionDir = "D:\Study\Datasets\AATEST\\instru_pulse\\"
# extensionDir = "D:\Study\Datasets\AATEST\\new_short\\"
#
# extensionDir = "D:\Study\Datasets\AAATEST\\"
imgx =   512#360#512#854#720#360#854#360
imgy = 384#288#384#480#288#480#288
fmt = ".avi"#".mp4"#".avi"  #.map4 is ori video suitable for everything except FN,.avi is resize suitable for everything
precision4given = 0.11#



fps = 30
# fps = 25
# videoname = "4"#"11"#"11"#"h3"#"3"#"cardNresp1"#"card1"#"WinB25"#"resp3"#"card1"# \"+videoname+"#BINW25  WINB127 WINB25
# givenfreq = 1.2#0.95#1.25#1.3#1.3#1.3#0.8#1.25#1.3#1.5#1.25#0.8#1.25#0.8#1#0.8#1.5#2.3# 0.8#0.35#1.5   # edit it to 1.1  the result is not good as expected!
# time_range = [0,5]#[1,8]#35 [21,31]#[0,20]
conv_times =1#2#2# 0 #(0,1,2,3)  0 mean no colv   when you set this to true, make sure the note is "ang"
numx_list = [512]#,256,128,64,32]
numy_list = [384]#,192,96,48,24]
mode_list = ["FN"]#,"GF","GF"]#,"FN","FN","HS","HS"]
note_list = ["mag"]#,"mag","ang"]#,"mag","ang","mag","ang"]
# mode_list = ["GF"]#,"GF","GF"]#,"FN","FN","HS","HS"]
# note_list = ["ang"]#,"mag","ang"]#,"mag","ang","mag","ang"]
mode_list = ["gray"]#,"GF","GF"]#,"FN","FN","HS","HS"]
note_list = [""]#,"mag","ang"]#,"mag","ang","mag","ang"]
video_list = []
givenfreq_list = []
time_range_list = []

video_list = [22,22,44,44,55,7,88,88,100,133,133,155,19,20,244]
givenfreq_list = [1.0,1.0,1.0,1.0,1.0,0.9,0.8,0.8,0.9,1.0,1.0,1.0,1.1,1.1,1.6]
time_range_list = [[1,6],[11,16],[1,6],[12,17],[10,15],[6,11],[1,6],[10,15],[4,9],[0,5],[16,21],[1,6],[0,5],[7,12],[0,5]]
#  there is issue for 10 not for 6 with the simple script
video_list = [10]#[6]
givenfreq_list = [1.2]#[1.6]
time_range_list = [[0,5]]#[[15,20]]


video_list = [22,22,44,44,55,7,88,88,100,133,133,155,19,20,244]
givenfreq_list = [1.0,1.0,1.0,1.0,1.0,0.9,0.8,0.8,0.9,1.0,1.0,1.0,1.1,1.1,1.6]
time_range_list = [[1,6],[11,16],[1,6],[12,17],[10,15],[6,11],[1,6],[10,15],[4,9],[0,5],[16,21],[1,6],[0,5],[7,12],[0,5]]

#maybe set more type?
#

video_list = [6]
givenfreq_list = [1.6]
time_range_list = [[15,20]]
video_list = [9]#[6]
givenfreq_list = [1.2]#[1.6]
time_range_list = [[0,5]]#[[15,20]]


# numx_list = [60]#,256,128,64,32]
# numy_list = [50]#,192,96,48,24]
# imgx = 60
# imgy = 50

#  find 10 s videos to improve quality
for videoname,givenfreq,time_range in zip(video_list,givenfreq_list,time_range_list):
    print(videoname,givenfreq,time_range)
    videoname = str(videoname)
    t = time_range[1] - time_range[0]
    unit = round(t / 5.0)
    top = 5 * unit  # 20#10#20  #keep right there, only take up the highest 20 ID into account
    # top = 1*unit
    top4harmonic = 20 * unit  # 25#25
    realfreq4samples = fps  # it is not the final real freq
    step = round(fps / realfreq4samples)
    real_sample_freq = float(fps) / step
    df = real_sample_freq / (t * real_sample_freq - 1)
    meansigarrayDIR = extensionDir + videoname + "\\"
    path4csv = meansigarrayDIR + videoname + ".csv"
    truthimg_path = meansigarrayDIR + "GTmask.png"
    dir4masksets = meansigarrayDIR  # can recursive search
    gau_vibe_mask_dir = meansigarrayDIR + "other\\"

    for mode, note in zip(mode_list, note_list):
        for gridnumx, gridnumy in zip(numx_list, numy_list):

            String0 = str(gridnumx) + "_" + str(gridnumy)
            String = str(gridnumx) + "_" + str(gridnumy) + "_" + str(top) + "_" + str(top4harmonic) + "_" + str(
                conv_times) + "_" + str(precision4given) + "_" + str(givenfreq)
            String3 = str(time_range[0]) + "s_" + str(time_range[1]) + "s_top" + str(top) + "_" + str(
                top4harmonic) + "_conv" + str(conv_times) + "_nbr" + str(precision4given) + "_given" + str(
                givenfreq)
            videopath = extensionDir + videoname + fmt

            meansigarrayPath = meansigarrayDIR + mode + "\\" + note + "\size" + String0 + "\\" + "SIGS_" + mode + "_" + str(
                time_range[0]) + "_" + str(time_range[1]) + note + ".npy"
            path_prefix = extensionDir + videoname + "\\" + mode + "\\" + note + "\size" + String0 + "\\" + String3
            masksetpath = path_prefix + "_maskSet.npy"
            maskset = np.zeros((3, gridnumy, gridnumx), np.bool)

            if gridnumx == 512 and note != "ang":
                pixelwisesig_generate(mode)  # CARRY OUT THIS FOR ONE TIME with pixel wise mose THEN COMMENT OUT
                # pass
            gridsig_generate()  # this is used when increase the grid size,the pixel wise sig to generate obj sig without import video ,OF computing etc.
            flag, df, realtotalNUM = fft_window(conv_times, 1, 1, top, top4harmonic, fps, givenfreq,
                                                precision4given,
                                                meansigarrayPath, realfreq4samples, path_prefix)  # , [0.1, 0.3])

# generateALLstats4video(imgx, imgy, truthimg_path, path4csv, dir4masksets,gau_vibe_mask_dir)







