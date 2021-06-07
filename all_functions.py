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
from HS_GF_FN_GRAY_func import HS,GF,GRAY,FN
from test import draw2pointsRAW_FFT,on_EVENT_LBUTTONDOWN
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import norm

import numpy.ma as ma


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

def test1(pos_Y_from_fft,precision4given,top10_ID,topharmonic_ID,df,given_freq,maskNfreqID_infoMat):  # try to reduce FN

    gridnumy = top10_ID.shape[1]
    gridnumx = top10_ID.shape[2]

    ID_expect = given_freq/df
    top10_ID_filter = np.where(top10_ID *df< given_freq+ precision4given , top10_ID, 0)#  to solve the issue of 1.3/2.6
    top10_ID_filter = np.where(top10_ID_filter *df> given_freq- precision4given , top10_ID_filter, 0)#  to solve the issue of 1.3/2.6
    mask2D =  np.any(top10_ID_filter,axis= 0)#.astype(np.float64)
    bestIDX = np.zeros([gridnumy,gridnumx])

    closest_layer_number = np.argsort(-np.abs(top10_ID_filter-ID_expect),axis=0)[-1]#just 4 non-mask region is useful!
# how to attach the mask effeciently?
    for y in range(gridnumy):
        for x in range(gridnumx):
            if mask2D[y,x]:
                bestIDX[y,x] = top10_ID_filter[closest_layer_number[y,x],y,x]
                secondHarmonicIDX = 2*bestIDX[y,x]
                thirdHarmonicIDX = 3*bestIDX[y,x]
                #
                if (int(secondHarmonicIDX) not in topharmonic_ID[:,y,x]) \
                        and(int((secondHarmonicIDX+1)) not in topharmonic_ID[:,y,x])\
                        and(int((secondHarmonicIDX-1)) not in topharmonic_ID[:,y,x]) :
                    mask2D[y, x] = False
                    bestIDX[y, x] = 0
                    continue
                if(int(thirdHarmonicIDX) not in topharmonic_ID[:,y,x]) \
                        and(int((thirdHarmonicIDX+1)) not in topharmonic_ID[:,y,x])\
                        and(int((thirdHarmonicIDX-1)) not in topharmonic_ID[:,y,x]) \
                        and (int((thirdHarmonicIDX + 2)) not in topharmonic_ID[:, y, x]) \
                        and (int((thirdHarmonicIDX - 2)) not in topharmonic_ID[:, y, x]):
                    mask2D[y, x] = False
                    bestIDX[y, x] = 0
                    continue


    # mask2D = maskNfreqID_infoMat[0].astype(bool)
    bestStr = np.zeros_like(mask2D).astype(complex)
    for y in range(gridnumy):
        for x in range(gridnumx):
            if mask2D[y, x]:
                bestID = int(bestIDX[y,x])
                bestStr[y,x] = pos_Y_from_fft[bestID,y,x]
    bestStr = np.abs(np.absolute(bestStr))





# vis the result after test1
    img = plt.imshow(bestIDX*df, cmap='hot', interpolation='nearest')
    plt.colorbar(img)
    plt.title("freq distribution after test1")
    plt.savefig(path_prefix+"freqmapI.png")#v"freqmapF.png"
    plt.show()

    img2 = plt.imshow(bestStr, cmap='hot', interpolation='nearest')
    plt.colorbar(img2)
    plt.title("\"periodicness\" magnitude distribution after test1")
    plt.savefig(path_prefix + "magmapI.png")
    plt.show()

    # plt.matshow(bestStr)
    # plt.show()






    maskNfreqID_infoMat[0] = mask2D.astype(np.uint32)    #mask
    maskNfreqID_infoMat[1] = bestIDX#index

    return maskNfreqID_infoMat,bestStr

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def test2(bestStr,maskNfreqID_infoMat,df): # require relative strong strength


    bestStr1D = bestStr.flatten()
    bestStr1D = -np.sort(-bestStr1D)


    pdf_X =[i  for i in range(len(bestStr1D))]

    params = np.polyfit(pdf_X,bestStr1D,10)
    func = np.poly1d(params)
    grad2 = func.deriv(2)
    bestStr1D_fit = func(pdf_X)
    grad2_fit = grad2(pdf_X)
    grad2_fit = grad2(pdf_X)/grad2_fit.max()  #normolize
    indices = np.array(np.where(grad2_fit<0.05))
    idx4thr = indices[0,1]#  get the "guai" point
    absStr = bestStr1D[idx4thr]

    if note  == "ang":
        idx4thr = ""
        absStr = 60


    # print(func,grad2,indices)
    print("thrshold idx: ",idx4thr," thr4str: ",absStr)
    # plt.plot(pdf_X,grad2_fit,"o")
    step = int(len(bestStr1D) / 200)
    plt.plot(pdf_X,bestStr1D_fit,alpha = 1, color = "g", linestyle = "--", markersize = 4, linewidth = 1,label = "fitting")
    plt.plot(pdf_X[::step],bestStr1D[::step],"x",label= "original",color = "b",alpha = 0.5)
    plt.plot(idx4thr, absStr, 'x',color = "red",label = "thrshold Point")
    plt.title("\"periodicness\" curves in order")
    plt.legend()
    plt.savefig(path_prefix+"curves.png")
    plt.show()

    plt.plot(pdf_X,grad2_fit)
    plt.title("2nd derivative for \"periodicness\" magnitude distribution")
    # plt.show()

    mask2D = np.where(bestStr>absStr,True,False)


    # for y in range(gridnumy):
    #     for x in range(gridnumx):
    #         if mask2D[y, x]:
    #
    #
    #             bestID = maskNfreqID_infoMat[1][y,x]
    #             bestStr = np.abs(np.absolute(pos_Y_from_fft[bestID,y,x]))
    #
    #
    #             if(np.abs(np.absolute(bestStr))<absStr):  # check the aboslute atrength, avoid FP noise, maybe introduce another thr?
    #                 mask2D[y, x] = False
    #                 maskNfreqID_infoMat[1, y, x] = 0
    #                 continue
    #
    #
    #
    #             nbrL = np.abs(np.absolute(pos_Y_from_fft[max(1,(bestID-2)):(bestID-1),y,x]))
    #             nbrR = np.abs(np.absolute(pos_Y_from_fft[max(1,(bestID+2)):(bestID+3),y,x]))
    #
    #             if ((nbrL>bestStr).any()== True) or ((nbrR>bestStr).any()== True ):
    #                 mask2D[y, x] = False
    #                 maskNfreqID_infoMat[1, y, x] = 0
    #                 continue






    mask2D = mask2D.astype(np.uint32)
    maskNfreqID_infoMat[0] = mask2D
    maskNfreqID_infoMat[1] = np.where(mask2D == False, int(0),maskNfreqID_infoMat[1])
    img = plt.imshow(maskNfreqID_infoMat[1]*df, cmap='hot', interpolation='nearest')
    plt.colorbar(img)
    plt.title("freq distribution of final binary mask")
    plt.savefig(path_prefix + "freqmapF.png")
    plt.show()
    bestStr_test2 = np.where(bestStr>absStr,bestStr,0)

    img2 = plt.imshow(bestStr_test2, cmap='hot', interpolation='nearest')
    plt.colorbar(img2)
    plt.title("\"periodicness\" magnitude distribution of final binary mask ")
    plt.savefig(path_prefix + "magmapF.png")
    plt.show()



    return maskNfreqID_infoMat


def fft_window(conv_times,visuliseFFT,visuliseRAW,top,top4harmonic,fps,givenfreq,precision4given,path4signal,infoMatPath, sample_freq ):  # 默认为half

    array_3d = np.load(path4signal)

    if conv_times:
        array_3d2 =(array_3d/array_3d.max())**conv_times
        array_3d = array_3d*array_3d2

    num4indexY =  array_3d.shape[1]
    num4indexX =  array_3d.shape[2]
    maskNfreqID_infoMat = np.zeros((2,num4indexY,num4indexX),np.uint32)


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


            #  there is a bug here!
            # when the block window is small
            #maybe delete this in the final to speed up
            if real_totalNbr//2 < 6:
                print("the number of sampled frames is too less! Please increase <time_range> or <realfreq4samples> in the main!")
                return False, df

    order_list = np.argsort(-np.abs(np.absolute(pos_Y_from_fft)),axis=0)

    top10_ID = order_list[1:(top+1)]
    topharmonic_ID = order_list[1:(top4harmonic+1)]
    _,bestStr = test1(pos_Y_from_fft,precision4given,top10_ID,topharmonic_ID, df, givenfreq, maskNfreqID_infoMat )


    # test2(bestStr, maskNfreqID_infoMat, df)

    #check freqMap and saving mask as IMG
    gridwidth = int(imgx / num4indexX)
    gridheight = int(imgy / num4indexY)
    mask_2d = maskNfreqID_infoMat[0]


    y_idx = np.nonzero(mask_2d)[0]
    x_idx = np.nonzero(mask_2d)[1]
    mask_2d = np.zeros((imgy, imgx))
    for y, x in zip(y_idx, x_idx):
        mask_2d[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask_2d = mask_2d.astype(np.float32) * 255

    mask_2dCP = mask_2d.copy()
    cv2.namedWindow("image")
    img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP = (mask_2dCP,gridwidth, gridheight,pos_Y_from_fft,visuliseFFT,visuliseRAW,df,path4signal,array_3d,maskNfreqID_infoMat[1])
   # this func can interatively show curve
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP)
    cv2.imshow("image", mask_2dCP)
    cv2.waitKey(0)
    mask_path = path_prefix+"mask.png"
    cv2.imwrite(mask_path, mask_2dCP)#mask_2d
    end = time.time()
    print('fft time for all windows: ' + str(end - start))
    np.save(infoMatPath,maskNfreqID_infoMat)#infomat now is maskNfreqID_infoMat
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
        imgy) + "\\" + "SIGS_" + mode + "_" + str(imgx) + "_" + str(imgy) + "_" + str(
        time_range[0]) + "_" + str(time_range[1]) + note + ".npy"
    newpath2sig = meansigarrayDIR + mode + "\\" + note + "\size" + str(gridnumx) + "_" + str(
        gridnumy) + "\\" + "SIGS_" + mode + "_" + str(gridnumx) + "_" + str(gridnumy) + "_" + str(
        time_range[0]) + "_" + str(time_range[1]) + note + ".npy"
    sig = np.load(path2sig)
    sig = sig.reshape(-1, gridnumy, int(imgy / gridnumy), gridnumx, int(imgx / gridnumx))
    newsig = sig.mean(axis=(2, 4))
    np.save(newpath2sig, newsig)













#params you may want to change
fps = 25
# 可以非整除
realfreq4samples = 25  # it is not the final real freq
fmt = ".avi"#".mp4"#".avi"  #.map4 is ori video suitable for everything except FN,.avi is resize suitable for everything
precision4given = 0.11#
top = 5#20#10#20  #keep right there, only take up the highest 20 ID into account
top4harmonic =25
extensionDir = "D:\Study\Datasets\extension\\"
extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\irreg_motion\\"
extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\"
# extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\\arti_videos\\"



# params change frequently
imgx =   512#360#512#854#720#360#854#360
imgy = 384#288#384#480#288#480#288
gridnumx =       32#512#32#512#32#64#512#64#512#32#64#512#256#64#512#256#128#64#32#512#45#360#512# 128#512#32#128#512#360#  512#128#32#128#512#32#128#512# 32#128#512#128#512#32#512#32#512#32# 256#512#32#64#  128#32#64#128#256#512#32#64#128#256#512#45#90#180#360#180#360#512#854#720#360#854#360#180#90#360#90#180#360#720#180#72#36#36#18#360#180#18#36#72
gridnumy =      24#384#24#384#24#48#384#48#384#24#48#384#192#48# 384#192#96#48#24#36#384#36#288#384#96#384# 24#96#384#288# 384#96#96#384# 24#96#384#24#96#384#96#384#24#384#2
videoname = "1"#"11"#"11"#"h3"#"3"#"cardNresp1"#"card1"#"WinB25"#"resp3"#"card1"# \"+videoname+"#BINW25  WINB127 WINB25
givenfreq =0.95#1.25#1.3#1.3#1.3#0.8#1.25#1.3#1.5#1.25#0.8#1.25#0.8#1#0.8#1.5#2.3# 0.8#0.35#1.5   # edit it to 1.1  the result is not good as expected!
time_range = [1,6]#[1,8]#35 [21,31]#[0,20]
conv_times =1#2#2# 0 #(0,1,2,3)  0 mean no colv   when you set this to true, make sure the note is "ang"
mode = "gray"# "HS"  #gray   HS  FN GF
note = ""#"mag"#"mag"#"mag"#"mag"# "ang" "mag"

#
t = time_range[1]-time_range[0]
step = round(fps /realfreq4samples)
real_sample_freq = float(fps) / step
df =  real_sample_freq/ (t*real_sample_freq - 1)
sigmoid4t =  1/(1 + np.exp(-(t/15)))  #definately bigger than (0.5,1)  # suppose out video 5s-20s
sigmoid4given = 1/(1 + np.exp(-givenfreq)) # (0.5,1)#positively related to givenfreq, at the same time restrict it in [0,1]
String0 = str(gridnumx)+"_"+str(gridnumy)
String = str(gridnumx)+"_"+str(gridnumy)+"_"+str(top)+"_"+str(top4harmonic)+"_"+str(conv_times)+"_"+str(precision4given)+"_"+str(givenfreq)

videopath = extensionDir+videoname+fmt
meansigarrayDIR = extensionDir+videoname+"\\"
meansigarrayPath = meansigarrayDIR+mode+"\\"+note+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+note+".npy"
infoMatDIR = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\\"
infoMatPath = infoMatDIR+String+"_"+"infoMat_mask_ID"+".npy"
# mask_path = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\Mask"+String+".png"  #  the best mask!
path_prefix = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\\"+String


#
# pixelwisesig_generate(mode)  # CARRY OUT THIS FOR ONE TIME with pixel wise mose THEN COMMENT OUT
# #generate grid sig
#
gridsig_generate()# this is used when increase the grid size,the pixel wise sig to generate obj sig without import video ,OF computing etc.
#
flag,df,realtotalNUM = fft_window(conv_times,1,1,top,top4harmonic,fps, givenfreq,precision4given,meansigarrayPath,infoMatPath, realfreq4samples )#, [0.1, 0.3])

# draw2pointsRAW_FFT(vispos_YX,vispos_YX2,realfreq4samples,t,mode,note,gridnumx,gridnumy,time_range,meansigarrayDIR,df,videoname)