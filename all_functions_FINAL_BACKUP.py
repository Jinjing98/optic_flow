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



def test_strategy4window(array_1d,df,given):#z = 6#test_strategy4window(array_3d_best12idx_value_freq[:,i,j],df,given)
    # edit mask layer6 of the array_3c
    if(array_1d[2]>2*array_1d[3]):
        array_1d[5] = 1

def test_strategy4windowGloInit(thr4str,array_3d,df,given):
    mask = np.array(array_3d[2] > thr4str * array_3d[3], dtype=int)
    array_3d[5] = mask  #

def test_strategy(thr4str,thr4df,path4array3d,df,given,flag):
    array_3d = np.load(path4array3d)
    mask2d = array_3d[5]   #initially all 0
    if flag == False:
        return
    mostcommon, times = stats.mode(array_3d[4], axis=None)
    print(mostcommon, times)
    np.save(path4array3d, array_3d)  #

# the input should be 1 channel gray image(0-255)          2d 0/1 array
def morph(mask_img,k1,k2,k3, i1,i2,i3):
    # mask = np.load(infoMatPath)[5]
    # mask_img = mask.astype(np.float32) * 255
    cv2.imshow("before2", mask_img)
    cv2.waitKey(0)  # 2 4 4   1 1 2
    kernel4E = np.ones((k1,k1), np.uint8)  # 去白点噪声，应该一直很small  可以跟随grid的大小调整，同大，或者为其2/3,1/2
    kernel4D = np.ones((k2,k2), np.uint8)  # thick to original+去黑点（孔洞）  密集的话应该来多轮or增大此kernel  或者这两个参数都略微提高结果最好！
    kernel4E2 = np.ones((k3,k3), np.uint8)  # before this, make sure u have kicked out all the white/black particles
    mask_img = cv2.erode(mask_img, kernel4E, iterations=i1)# 这个迭代次数不能动，只能调整kernel大小，否则正确的也erode太厉害
    cv2.imshow("Ero2", mask_img)
    cv2.waitKey(0)
    mask_img = cv2.dilate(mask_img, kernel4D, iterations=i2)  # 密集的话应该这里来多轮迭代or增大此kernel4d  或者这两个参数都略微提高结果最好！
    cv2.imshow("Dil2", mask_img)
    cv2.waitKey(0)
    mask_img = cv2.erode(mask_img, kernel4E2, iterations=i3)
    cv2.imshow("final2", mask_img)
    cv2.waitKey(0)
    return mask_img
    # mask = (mask_img / 255.0).astype(np.int32)










#if you are sure it is big organ mode, by seeting the flag to true can make the result more convincing!
#if you have given approxmate freq, by setting given to the non zero value, can make the result more convincing!
def test_strategy4windowGlo(thr4str,thr4df,array_3d,df,given):#z = 6#test_strategy4window(array_3d_best12idx_value_freq[:,i,j],df,given)
    # edit mask layer6 of the array_3c


    if given:
        mask1 = np.array((array_3d[4] < given+thr4df*df), dtype=int)
        mask2 = np.array((array_3d[4] > given-thr4df*df), dtype=int)
        mask3 = np.array((mask1 == 1) & (mask2 == 1), dtype=int)
    else:
        mask3 = array_3d[5]
    #print(array_3d[4,])#float 64
    #print(array_3d[4,1,1:5])

    #find the index of most common freq in layer 0(strongest freq)
    a = array_3d[0].reshape(-1)
    counts = np.bincount(a.astype(np.int32))
    mostidx = np.argmax(counts)
    orderofmost = np.argsort(-np.abs(counts))
    print("the most frequent frequncy index : ",orderofmost[0])
    #the same result as above, but here it also count the times of the appearing
    mostfreq,times = stats.mode(array_3d[4], axis=None)
    print(mostfreq,times)
    print(array_3d[0],array_3d[1])
    # if mostcommon < given - 2*df or mostcommon > given + 2*df:
    #     print("the periodic is relatively weak to be confidently detected! Find another ")
    #     return False
    #print(array_3d[5])
    print(array_3d.shape)# 6 48 72
    print(array_3d[2].shape) #48,72
    print(array_3d[2])
    print(thr4str*array_3d[3])# this times is not good !
    print(df)
    #print(array_3d[2]>2*array_3d[3])# or + df?
    mask4 = np.array(array_3d[2]>thr4str*array_3d[3],dtype=int)#INT32
    mask = np.array((mask3 == 1) & (mask4 == 1), dtype=int)
    #print(mask == mask4)
    array_3d[5] = mask
    array_3d[6] = mask  # we don't need layer6 anymore, the latest version decided to do morph not for the array, but for extended gray img


# # processing “mask" again!
#     #形态学操作去噪声 去孔洞
#     mask_img = mask.astype(np.float32) * 255
#     # mask_img = cv2.cvtColor(mask.astype(np.float32) * 255, cv2.COLOR_GRAY2BGR)
#     cv2.imshow("before", mask_img)
#     cv2.waitKey(0)  #2 4 4   1 1 2
#     kernel4E = np.ones((2, 2), np.uint8)#去白点噪声，应该一直很小
#     kernel4D = np.ones((4, 4), np.uint8)#thick to original+去黑点（孔洞）  密集的话应该来多轮or增大此kernel  或者这两个参数都略微提高结果最好！
#     kernel4E2 = np.ones((4, 4), np.uint8)#before this, make sure u have kicked out all the white/black particles
#     mask_img = cv2.erode(mask_img, kernel4E, iterations=1)
#     cv2.imshow("Ero", mask_img)
#     cv2.waitKey(0)
#     mask_img = cv2.dilate(mask_img,kernel4D,iterations=5)#密集的话应该这里来多轮迭代or增大此kernel4d  或者这两个参数都略微提高结果最好！
#     cv2.imshow("Dil", mask_img)
#     cv2.waitKey(0)
#     mask_img = cv2.erode(mask_img, kernel4E2, iterations=2)
#     cv2.imshow("final",mask_img)
#     cv2.waitKey(0)

    #mask = np.uint8(mask)
    #  change it to mask, the result get improved, as you can see the boundary is not wrongly regarded as info region
    #array([ True, False,  True], dtype=bool)



def visual_save_mask(mask_path,maskNfreqID_infoMat,imgx,imgy,numx,numy):
    gridwidth = int(imgx / numx)
    gridheight = int(imgy / numy)
    mask_2d = maskNfreqID_infoMat[0]

    y_idx = np.nonzero(mask_2d)[0]
    x_idx = np.nonzero(mask_2d)[1]
    mask_2d = np.zeros((imgy, imgx))
    for y, x in zip(y_idx, x_idx):
        mask_2d[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask_2d = mask_2d.astype(np.float32) * 255

    cv2.imshow("",mask_2d)
    cv2.waitKey(0)
    cv2.imwrite(mask_path,mask_2d)




def morph_visul_savemask(maskDir,mask_path,mask_pathNO,path4array,imgx,imgy,numx,numy,k1,k2, k3, i1, i2, i3):
    gridwidth = int(imgx/numx)
    gridheight = int(imgy/numy)
    mask_2d = np.load(path4array)[5]
    print(mask_2d.shape)#12 18
    print(mask_2d)
    # print(np.nonzero(mask_2d))
    #print(np.nonzero(mask_2d)[0])#189
    y_idx = np.nonzero(mask_2d)[0]
    x_idx = np.nonzero(mask_2d)[1]
    mask_2d = np.zeros((imgy,imgx))
    for y,x in zip(y_idx, x_idx):
            mask_2d[y*gridheight:(y+1)*gridheight,x*gridwidth:(x+1)*gridwidth] = 1
    #cv2.imshow("",mask_2d)
    #cv2.waitKey(0)
    # return np.uint8(mask_2d)

    # mask = np.load(infoMatPath)[5]
    # k1, k2, k3, i1, i2, i3 = 5, 12 ,4, 1, 4 ,2#erotion(no white points denoise) dilation(no black points) erosion
    # k1 = int(gridwidth*2/3)  #

    mask_img = mask_2d.astype(np.float32) * 255
    morph_mask_img = morph(mask_img, k1,k2,k3,i1,i2,i3)#k1,k1,k3, i1,i2,i3
    mask_2d = (morph_mask_img / 255.0).astype(np.int32)
    mask_2dNO = (mask_img / 255.0).astype(np.int32)
    # mask_2dNO = np.load(path4array)[6]
    # y_idxNO = np.nonzero(mask_2dNO)[0]
    # x_idxNO = np.nonzero(mask_2dNO)[1]
    # mask_2dNO = np.zeros((imgy, imgx))
    # for y, x in zip(y_idxNO, x_idxNO):
    #     mask_2dNO[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1

    # mask2d_gray = cv2.cvtColor(mask_2d.astype(np.float32)*255, cv2.COLOR_GRAY2BGR)
    #cv2.imwrite(mask_path, morph_mask_img)
    cv2.imwrite(mask_pathNO, mask_img)








    # mask_2dNO = np.load(path4array)[6]
    # y_idxNO = np.nonzero(mask_2dNO)[0]
    # x_idxNO = np.nonzero(mask_2dNO)[1]
    # mask_2dNO = np.zeros((imgy, imgx))
    # for y, x in zip(y_idxNO, x_idxNO):
    #     mask_2dNO[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    #
    # # mask2d_gray = cv2.cvtColor(mask_2d.astype(np.float32)*255, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite(mask_path,mask_2d.astype(np.float32)*255)
    # cv2.imwrite(mask_pathNO,mask_2dNO.astype(np.float32) * 255)

    return mask_2d,mask_2dNO

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
            cv2.imwrite(outimgpathNO,frameNO)

            break








#visulmask("D:\Study\\flownet2-pytorch\infoMat_6height.npy",360,288,20,24)#360 288



# T here is the range relative to the .npy file.  (not relative to the video, for the case of .npy start with 0, the two are the same)
# carry out fft.
# from avoid FN point of view, more general
def test1(top10_ID,topharmonic_ID,df,given_freq,maskNfreqID_infoMat,thr4nbr):  # try to reduce FN
    #  all true if it is the wider neighbor of given
    # how to define "a good neighbour?" how to set thr4nbr?
    # thr4nbr = 1


    gridnumy = top10_ID.shape[1]
    gridnumx = top10_ID.shape[2]

    ID_expect = given_freq/df
    top10_ID_filter = np.where(top10_ID *df< given_freq+ 0.11 , top10_ID, 0)#  to solve the issue of 1.3/2.6
    # top10_ID = np.where(top10_ID <  ID_expect*2 + thr4nbr , top10_ID, 0)#  to solve the issue of 1.3/2.6
    top10_ID_filter = np.where(top10_ID_filter *df> given_freq- 0.11 , top10_ID_filter, 0)#  to solve the issue of 1.3/2.6

    # top10_ID = np.where(top10_ID >  ID_expect - thr4nbr , top10_ID, 0)

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


    plt.imshow(bestIDX*df, cmap='hot', interpolation='nearest')
    plt.show()


    maskNfreqID_infoMat[0] = mask2D.astype(np.uint32)    #mask
    maskNfreqID_infoMat[1] = bestIDX#index


    #harmonic test::this should be done before test 2, a flaw in test 2, the thrshold is depending on the  mag of unmasked ones, unmasked ones depends on the distribution of test1
    # the result of test 1 should reduce as much as noise possib;e


    return maskNfreqID_infoMat

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def test2(absStr,pos_Y_from_fft,top10_ID,maskNfreqID_infoMat,df,given_freq,thr4str): # require relative strong strength
    mask2D = maskNfreqID_infoMat[0].astype(bool)
    Boundary_thrSma_thrBig = 2*thr4nbr   # THE UPPPER BOUND FOR deltaID_BEST_EXPECT
    ID_expect = given_freq / df

    # array_3d2 = np.load(path4signal_magcoeef)
    # array_3d2 /= array_3d2.max()
    # array_3d = array_3d * array_3d2 * array_3d2
    # bestID_mat = maskNfreqID_infoMat[1]
    # array3d = np.zeros((62,384,512))
    # array3d[bestID_mat] = 1
    # mx = ma.masked_array(pos_Y_from_fft, mask2D=[0, 0, 0, 1, 0])
    # maxStr = np.abs(np.absolute(pos_Y_from_fft[, y, x])).max()

    bestStr = np.zeros_like(mask2D).astype(complex)
    for y in range(gridnumy):
        for x in range(gridnumx):
            if mask2D[y, x]:


                bestID = maskNfreqID_infoMat[1][y,x]
                # bestStr = np.abs(np.absolute(pos_Y_from_fft[bestID,y,x]))
                bestStr[y,x] = pos_Y_from_fft[bestID,y,x]
    bestStr = np.abs(np.absolute(bestStr))
    max_bestStr = bestStr.max()
    plt.imshow(bestStr, cmap='hot', interpolation='nearest')
    plt.show()
    # plt.matshow(bestStr)
    # plt.show()

    bestStr1D = bestStr.flatten()
    bestStr1D = -np.sort(-bestStr1D)


    pdf_X =[i  for i in range(len(bestStr1D))]

    params = np.polyfit(pdf_X,bestStr1D,10)
    func = np.poly1d(params)
    # func_grad1 = np.gradient(func)
    grad2 = func.deriv(2)



    bestStr1D_fit = func(pdf_X)
    grad2_fit = grad2(pdf_X)
    grad2_fit = grad2(pdf_X)/grad2_fit.max()  #normolize
    indices = np.array(np.where(grad2_fit<0.05))
    idx4thr = indices[0,1]
    absStr = bestStr1D[idx4thr]

    # print(func,grad2,indices)
    print("idx: ",idx4thr," thr4str: ",absStr)
    # plt.plot(pdf_X,grad2_fit,"o")

    # alpha = 0.5, color = "r", linestyle = "--", marker = "x", markersize = 4, linewidth = 1, label = 'NN'


    plt.plot(pdf_X,grad2_fit)
    plt.show()
    plt.plot(pdf_X,bestStr1D_fit,alpha = 1, color = "g", linestyle = "--", markersize = 4, linewidth = 1,label = "fitting")
    # plt.plot(pdf_X[::1000],bestStr1D[::1000], alpha = 0.5,color = "b", marker = "x", markersize = 4,label= "original")


    # plt.plot(pdf_X,bestStr1D_fit,"r",label = "fitting",color = "blue")
    step = int(len(bestStr1D)/200)
    plt.plot(pdf_X[::step],bestStr1D[::step],"x",label= "original",color = "b",alpha = 0.5)
    plt.plot(idx4thr, absStr, 'o',color = "red",label = "thrshold Point")
    plt.legend()
    plt.show()


    for y in range(gridnumy):
        for x in range(gridnumx):
            if mask2D[y, x]:


                bestID = maskNfreqID_infoMat[1][y,x]
                bestStr = np.abs(np.absolute(pos_Y_from_fft[bestID,y,x]))


                if(np.abs(np.absolute(bestStr))<absStr):  # check the aboslute atrength, avoid FP noise, maybe introduce another thr?
                    mask2D[y, x] = False
                    maskNfreqID_infoMat[1, y, x] = 0
                    continue
                # else:  # since I used top20, this else is quite necessary!  since we may have some points here
                #     topID = top10_ID[0, y, x]
                #     topStr = np.abs(np.absolute(pos_Y_from_fft[topID, y, x]))
                #     deltaID_BEST_EXPECT= np.abs(bestID-ID_expect)  # depend on if the distance of bestID to
                #                                                     # expect(given_ID), we use different thrStr
                #                                                     #if bigger distance,then less possibility of Positive True window,
                #                                                     # so we use a stricter(bigger)  thr4str
                #     bound = np.min([thr4nbr/4,2])  # 4<5 lower bound of thr4nbr   (5,15)
                #     thr4str =[thr4strSma if deltaID_BEST_EXPECT<bound else thr4strBig]  # since here we already passed the strengeth test, so the thr4strSma can be quite small [0.05-0.1]
                #     # thr4str = 0.4
                #     if (bestStr / topStr < thr4str):  #avoid FP periodic region with other freq(since in test1 we have a quite wider neighbor)
                #         mask2D[y, x] = False
                #         maskNfreqID_infoMat[1, y, x] = 0

                #
                # elif(nLStr>bestStr or nRstr>bestStr):
                #     mask2D[y, x] = False
                #     maskNfreqID_infoMat[1, y, x] = 0
                #     continue
                # else:#  this part is bad, failed in GF3, OR THIS IS THE ISSUE OF VIDEO3, THE MOST STRONG SIG IS HIGH 2.6
                #     topID = top10_ID[0, y, x]
                #     topStr = np.abs(np.absolute(pos_Y_from_fft[topID, y, x]))
                #     if((topID-bestID)>2*thr4nbr/3 and topStr>bestStr*thr4str):#this case we suppose is points periodic motion with another freq(not close enough to given to be recognized as given)
                #         mask2D[y, x] = False
                #         maskNfreqID_infoMat[1, y, x] = 0


                # peak detection:

                nLStr = np.abs(np.absolute(pos_Y_from_fft[max(1,bestID-1),y,x]))
                nRstr = np.abs(np.absolute(pos_Y_from_fft[bestID+1,y,x]))
                nLStrr = np.abs(np.absolute(pos_Y_from_fft[max(1,bestID-2), y, x]))
                nRstrr = np.abs(np.absolute(pos_Y_from_fft[bestID + 2, y, x]))
                nbr = np.array([nLStr,nRstr])

                #considering that the estimation of greq have 1/5 = 0.2hz error,which also means one nbr error,<>
                nbrL = np.abs(np.absolute(pos_Y_from_fft[max(1,(bestID-2)):(bestID-1),y,x]))
                nbrR = np.abs(np.absolute(pos_Y_from_fft[max(1,(bestID+2)):(bestID+3),y,x]))

                # if y==11 and x == 14:
                #     print()
                if ((nbrL>bestStr).any()== True) or ((nbrR>bestStr).any()== True ):
                # if   (nbr>bestStr).any()== True :
                    mask2D[y, x] = False
                    maskNfreqID_infoMat[1, y, x] = 0
                    continue






    mask2D = mask2D.astype(np.uint32)
    maskNfreqID_infoMat[0] = mask2D
    return maskNfreqID_infoMat





#将两个定义为同一个T!!!! 一旦换头，应from scratch  .npy!
#meansigarrayPath,String0,infoMatPath, realfreq4samples,vispos_YX
def fft_window(conv_times,absStr,visuliseFFT,visuliseRAW,top,top4harmonic,fps,givenfreq, thr4nbr,thr4str,T,path4signal,path4signal_magcoeef,String0,infoMatPath, sample_freq,vispos_YX = [10,20]):  # 默认为half
    # path4signal = DIR4signal+mode+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"



    idx_start = T[0]*fps
    idx_end = T[1]*fps
    array_3d = np.load(path4signal)

  # this will make ang mode result better:  it use normolized mag as coeef for ang!
    if conv_times:
        # array_3d2 = np.load(path4signal_magcoeef)
        array_3d2 =(array_3d/array_3d.max())**conv_times
        # array_3d = array_3d*array_3d2*array_3d2*array_3d2
        array_3d = array_3d*array_3d2

    num4indexY =  array_3d.shape[1]
    num4indexX =  array_3d.shape[2]
    top10_ID = np.ones((top,num4indexY,num4indexX))  #  how to choose this 5?
    maskNfreqID_infoMat = np.zeros((2,num4indexY,num4indexX),np.uint32)


    # array_3d_best12idx_value_freq = np.ones((7,num4indexY,num4indexX))  # index1,2 value1,2 freq 1,2 for most n seconded strong freq+ layer7: noisy mask without too much checking!
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
            # order_list = np.argsort(-np.abs(pos_Y_from_fft))

            #  there is a bug here!
            # when the block window is small
            #maybe delete this in the final to speed up
            if real_totalNbr//2 < 6:
                print("the number of sampled frames is too less! Please increase <time_range> or <realfreq4samples> in the main!")
                return False, df


    #
    # if visuliseFFT:
    #     Y = vispos_YX[0]
    #     X = vispos_YX[1]
    #     M = pos_Y_from_fft[:,Y,X].size
    #     ydebug = np.absolute(pos_Y_from_fft[:,Y,X])
    #     testkk = np.abs(np.absolute(pos_Y_from_fft[:,Y,X]))
    #     testll = (np.all(testkk)==0)
    #     test = np.argsort(-np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))
    #     f = [df * n for n in range(0, M)]
    #
    #     # pl.semilogy(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))#?
    #     pl.plot(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))
    #     pl.xlabel('freq(Hz)')
    #     pl.title("positiveHalf fft YX" + str(Y) + "_" + str(X))
    #     pl.savefig(path4signal[:-4] +"_"+ str(Y)+"_"+str(X)+".png")
    #     pl.show()

    # order_list = np.zeros(pos_Y_from_fft.shape,dtype=np.uint32)  # this section can not order 0 well, but the first statge of  test2 can hadnle it
    # a = np.abs(np.absolute(pos_Y_from_fft))
    # order_list = [8 if a.any(axis = 0) else 0]
    order_list = np.argsort(-np.abs(np.absolute(pos_Y_from_fft)),axis=0)
    # testkk = np.abs(np.absolute(pos_Y_from_fft[:, Y, X]))




    top10_ID = order_list[1:(top+1)]
    topharmonic_ID = order_list[1:(top4harmonic+1)]
    test1(top10_ID,topharmonic_ID, df, givenfreq, maskNfreqID_infoMat, thr4nbr)
    a = maskNfreqID_infoMat.copy()
    test2(absStr,pos_Y_from_fft,top10_ID,maskNfreqID_infoMat,df,givenfreq,thr4str)
    improve = a[0].astype(np.uint8)-maskNfreqID_infoMat[0].astype(np.uint8)

    #check freqMap and saving mask as IMG
    gridwidth = int(imgx / num4indexX)
    gridheight = int(imgy / num4indexY)
    mask_2d = maskNfreqID_infoMat[0]
    freqMap = maskNfreqID_infoMat[1]*df
    # print("the freqMap of the mask: ", freqMap)


    y_idx = np.nonzero(mask_2d)[0]
    x_idx = np.nonzero(mask_2d)[1]
    mask_2d = np.zeros((imgy, imgx))
    for y, x in zip(y_idx, x_idx):
        mask_2d[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask_2d = mask_2d.astype(np.float32) * 255

    # cv2.imshow("", mask_2d)
    # cv2.waitKey(0)
    # cv2.imwrite(mask_path, mask_2d)
    mask_2dCP = mask_2d.copy()
    cv2.namedWindow("image")
    img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP = (mask_2dCP,gridwidth, gridheight,pos_Y_from_fft,visuliseFFT,visuliseRAW,df,path4signal,array_3d,maskNfreqID_infoMat[1])
   # this func can interatively show curve
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP)
    cv2.imshow("image", mask_2dCP)



    # if visuliseFFT:
    #     Y = vispos_YX[0]
    #     X = vispos_YX[1]
    #     M = pos_Y_from_fft[:,Y,X].size
    #
    #     f = [df * n for n in range(0, M)]
    #
    #     # pl.semilogy(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))#?
    #     pl.plot(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))
    #     pl.xlabel('freq(Hz)')
    #     pl.title("positiveHalf fft YX" + str(Y) + "_" + str(X))
    #     pl.savefig(path4signal[:-4] +"_"+ str(Y)+"_"+str(X)+".png")
    #     pl.show()



    cv2.waitKey(0)
    cv2.imwrite(mask_path, mask_2dCP)#mask_2d
















    # test_strategy4windowGlo(thr4str,thr4df,array_3d_best12idx_value_freq,df,givenfreq)
    end = time.time()
    print('fft time for all windows: ' + str(end - start))
    np.save(infoMatPath,maskNfreqID_infoMat)#infomat now is maskNfreqID_infoMat
    return True,df,real_totalNbr
def GF_window(videopath,sigDIR,mode,String0,ranges,gridnumX,gridnumY ):  # choose ranges given a video length to collect npy
    #meansigarrayPath = meansigarrayDIR+mode+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"
    sigpath = sigDIR+mode+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"



#HS(videopath,flow_path,sigpath,range)


    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    _, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    lower = fps * ranges[0]
    upper = fps * ranges[1]
    framenum = upper - lower

    hsv_mask = np.zeros_like(frame1)
    hsv_mask[..., 1] = 255

    gridWidth = int(width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
    gridHeight = int(height / gridnumY)  # 240
    SIGS_gray = np.zeros((framenum, gridnumY, gridnumX))  # 4,3
    SIGS_ang = np.zeros((framenum, gridnumY, gridnumX))
    SIGS_mag = np.zeros((framenum, gridnumY, gridnumX))

    i = 1
    timepoint = 0
    start1 = time.time()
    while (i):

        ret, frame2 = cap.read()
        if i < lower:  # 直接从好帧开始运行
            i += 1
            continue
        if i >= upper:
            break
        if ret != True:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        start = time.time()
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=5, iterations=5,
                                            poly_n=5,
                                            poly_sigma=1.1, flags=0)
        end = time.time()
        print("flow computing time: " + str(end - start))

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)  # in radians not degrees
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255,
                                         cv2.NORM_MINMAX)  # nolinear transformation of value   sigmoid?
        # var = hsv_mask[..., 2] - mag

        start = time.time()
        # before it took around 3 mins, now it is way faster based on numpy vectorization
        # https://codingdict.com/questions/179693
        next = next.reshape(gridnumY, gridHeight, gridnumX, gridWidth)
        next_ang = hsv_mask[..., 0].reshape(gridnumY, gridHeight, gridnumX, gridWidth)
        next_mag = hsv_mask[..., 2].reshape(gridnumY, gridHeight, gridnumX, gridWidth)
        # print(next[0,:,0,:])

        SIGS_gray[timepoint] = next.mean(axis=(1, 3))
        SIGS_ang[timepoint] = next_ang.mean(axis=(1, 3))
        SIGS_mag[timepoint] = next_mag.mean(axis=(1, 3))

        end = time.time()
        print('time to compute mean signal for windows' + str(end - start))
        timepoint += 1
        i += 1
    end1 = time.time()
    print("total time to collect mean gray and mean flow: " + str(end1 - start1))
    print(np.array(SIGS_ang).size)
    print(np.array(SIGS_mag).size)
    # np.save(sigpath+"GFSIGS_ang.txt",SIGS_ang)
    # np.save(sigpath+"GFSIGS_mag.txt",SIGS_mag)
    np.save(sigpath, SIGS_gray)
    # np.save("D:\\Study\\Datasets\\AEXTENSION\\Cho80_extension\\static_cam\\pulseNstatic\\1\\gray\\size854_480\\SIGS_gray854_480.npy",SIGS_gray)



fmt = ".avi"#".mp4"#".avi"  #.map4 is ori video suitable for everything except FN,.avi is resize suitable for everything

# 384#24#192#384#24#48 #   96#24#48#96#192#384#24#48#96#192#384#36#72#144#288#144#288#384#480#288#480#288#144#36#144#18#36#72#144#288#144#48#24#24#12#288#144#12#24#48
# gridwidth =  int(imgx/gridnumx)
# gridheight =int(imgy/gridnumy)

# thr4nbr = 6# if t is small, this shoudl be small too
# thr4str = 0.3#1.2  bigger stricter   1.0  make no sense

fps = 25
# 可以非整出，非整除不会出错！?
realfreq4samples = 25  # it is not the final real freq


# vispos_YX = #[192,200]#[130,80]  #the pos of the grid not the pixel!

# this is gird coordinate not pixel coord!! the two are the same when pxiel wise grid!
vispos_YX =   [200,180] #   pulse
vispos_YX2 = [10,70]# non pulse[1,4]#

imgx =   512#360#512#854#720#360#854#360
imgy =   384#288#384#480#288#480#288
gridnumx =  512#360#  512#128#32#128#512#32#128#512# 32#128#512#128#512#32#512#32#512#32# 256#512#32#64#  128#32#64#128#256#512#32#64#128#256#512#45#90#180#360#180#360#512#854#720#360#854#360#180#90#360#90#180#360#720#180#72#36#36#18#360#180#18#36#72
gridnumy =    384#288# 384#96#96#384# 24#96#384#24#96#384#96#384#24#384#2
time_range = [0,15]#[1,8]#35 [21,31]#[0,20]
givenfreq =  1.3#1.3#1.3#1.3#0.8#1.25#1.3#1.5#1.25#0.8#1.25#0.8#1#0.8#1.5#2.3# 0.8#0.35#1.5   # edit it to 1.1  the result is not good as expected!
videoname = "h3"#"11"#"11"#"h3"#"3"#"cardNresp1"#"card1"#"WinB25"#"resp3"#"card1"# \"+videoname+"#BINW25  WINB127 WINB25
conv_times =2#2#2# 0 #(0,1,2,3)  0 mean no colv   when you set this to true, make sure the note is "ang"
mode = "FN"# "HS"  #gray   HS  FN GF
note = "mag"#"mag"#"mag"#"mag"#"mag"# "ang" "mag"
absStr =20000#2#4.5#4.5#
thr4nbr = 1.1#np.around( 2*sigmoid4t*sigmoid4given/(0.5*0.5),2) #0.5 is the sigmoid4t when t == 0 #ΔHZ = thr4nb*df.  It is reasonable to let ΔHZ be positively related to givenfreq
# thr4nbr = max(min(15, thr4nbr), 5)  # restric thr4nbr to [3,10]  #quite genral of test1, we try to reduce FN but also introduce some FP
thr4nbr = min(6,thr4nbr)

t = time_range[1]-time_range[0]
step = round(fps /realfreq4samples)
real_sample_freq = float(fps) / step

df =  real_sample_freq/ (t*real_sample_freq - 1)
sigmoid4t =  1/(1 + np.exp(-(t/15)))  #definately bigger than (0.5,1)  # suppose out video 5s-20s
sigmoid4given = 1/(1 + np.exp(-givenfreq)) # (0.5,1)#positively related to givenfreq, at the same time restrict it in [0,1]

top = 5#20#10#20  #keep right there, only take up the highest 20 ID into account
top4harmonic =25

# this value should be bigger for irre case. such as 15
#ang 20  mag 5.5 gray
#3#6#3#5#30#20#8#20#30#30#10#2#5#5#5#10#20#5.5#20#10.5#.5#15#15#10.5  #final!set this to 5 is nice for staticpulse video! how to set the parms?  to deal with the case where, this vlaue was 0.5, it should be bigger for noisy irre image?
# thr4strSma = 0.05
# thr4strBig = 0.6#suggest < 1  #  since our thr4nbr is quite general(big), here we have to be strict for this thr4strBig to further filtered out periodic sig but with another freq(wider neighbr of given but not given)


thr4str = 5#6#6#6#6#4#3#2
String0 = str(gridnumx)+"_"+str(gridnumy)
String = str(gridnumx)+"_"+str(gridnumy)+"_"+str(thr4nbr)+"_"+str(top)+"_"+str(thr4str)+"_"+str(absStr)+"_"+str(givenfreq)

 # THIS IS JUST FOR fft signal source, both ang and mag sigs have been generated simutanously in the last step

extensionDir = "D:\Study\Datasets\extension\\"
extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\irreg_motion\\"
extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\"
# extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\\arti_videos\\"
videopath = extensionDir+videoname+fmt
meansigarrayDIR = extensionDir+videoname+"\\"
meansigarrayPath = meansigarrayDIR+mode+"\\"+note+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+note+".npy"
meansigarrayPath_magcoeef = meansigarrayDIR+mode+"\\"+"mag"+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+"mag"+".npy"
# meansigarrayPath_magcoeef = meansigarrayDIR+mode+"\\"+"ang"+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+"ang"+".npy"
#D:\\Study\\Datasets\\AEXTENSION\\Cho80_extension\\static_cam\\pulseNstatic\\video17 00_00_27-00_00_36\\gray\\size854_480\\SIGS_gray854_480.npy'

infoMatDIR = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\\"
infoMatPath = infoMatDIR+ "infoMat2_mask_ID"+String+"_"+str(realfreq4samples)+note+".npy"

mask_path = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\Mask"+String+".png"  #  the best mask!
mask_img_path = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\mask_img"+String+".png"
mask_pathNO = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\Mask"+String+"NO.png"
mask_img_pathNO = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String0+"\mask_img"+String+"NO.png"
maskDir = extensionDir+videoname+"\\"+mode+"\\"+note+"\size"+String+"\\"

#(videopath,sigDIR,String0,ranges,gridnumX,gridnumY)
# GF_window(videopath, meansigarrayDIR,mode,String0, time_range, gridnumx,gridnumy )

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

#
pixelwisesig_generate(mode)  # CARRY OUT THIS FOR ONE TIME with pixel wise mose THEN COMMENT OUT
# #generate grid sig
#
# gridsig_generate()# this is used when increase the grid size,the pixel wise sig to generate obj sig without import video ,OF computing etc.
#
# flag,df,realtotalNUM = fft_window(conv_times,absStr,1,1,top,top4harmonic,fps, givenfreq, thr4nbr,thr4str,time_range,meansigarrayPath,meansigarrayPath_magcoeef,String0,infoMatPath, realfreq4samples,vispos_YX)#, [0.1, 0.3])


# draw2pointsRAW_FFT(vispos_YX,vispos_YX2,realfreq4samples,t,mode,note,gridnumx,gridnumy,time_range,meansigarrayDIR,df,videoname)