import numpy as np
import pylab as pl
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import butter, lfilter
import time
import pylab as plt
from scipy import stats



#考虑滑动均值滤波和带通滤波进一步优化  可视化频率   （已知区间  PCA  可信度衡量 统计学

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def fft(fs,T,path, real_freq,flag = 'half',bandpass = False,Range = None,):   #  默认为half




    #jinjing


    # 采样频率
    # fs=25   #  反映在我们的应用中是视频帧率
    # totalT = 25 # 单位为s 30 25
    totalT = (T[1]-T[0])

    totalNbr = fs * totalT

    #jinjing test
    real_totalNbr = real_freq * totalT


    # 采样步长
    # t = [x/fs for x in range(totalNbr)]

    """
    设计的采样值
    假设信号y由4个周期信号叠加所得,如下所示
    """
    # y = np.load('D:\Study\Datasets\\Cam\GFflow\\SIGS_mag.txt.npy')[0:totalNbr]
    # y = np.load('D:\Study\Datasets\\Cam\HSflow\\SIGS_ang.txt.npy')[0:totalNbr]
    # y = np.load('D:\Study\Datasets\\Cam\\NNflow\\SIGS_ang.txt.npy')[0:totalNbr]
    y = np.load(path)[0:totalNbr]
    y = np.load(path)[T[0]*fs:T[1]*fs]
    y = y[::int((fs//real_freq))]   #   60/real_freq
    # y = np.load(path)[24*30:24*30+totalNbr]   #debug
    t = [totalT*x/y.size for x in range(y.size)]


    # print(y)
    # print(y.size)
    # y = [ 3.0 * np.cos(2.0 *np.pi * 0.50 * t0  )
    #      # + 1.5 * np.cos(800* t0 + np.pi * 90/180)
    #      # +  1.0 * np.cos(2.0 * np.pi * 800 * t0 + np.pi * 120/180)
    #      # +  2.0 * np.cos(1/2.0 /np.pi * 220 * t0 + np.pi * 30/180)
    #      for t0 in t ]




    # pl.plot(t,y)   #   y is orignal signal
    # pl.xlabel('time(s)')
    # pl.title("original signal")
    # pl.show()

    """
    现在对上述信号y在0-1秒时间内进行频谱分析，
    
    本案例中采样频率为1048Hz,即单位时间内采样点数为1048
    """
    # # 采样点数
    # N=len(t) ie totalNbr
    # # 采样频率
    # fs=25.0
    # # 分辨率
    df = real_freq/(real_totalNbr-1)   #0.03HZ
    # # 构建频率数组
    # f = [df*n for n in range(0,totalNbr)]   #  最大频率为30HZ

    # f = np.fft.fftfreq(real_totalNbr, d=totalT)
    Y = np.fft.fft(y)*2/real_totalNbr  #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
    # absY = [np.abs(x) for x in Y]      #求傅里叶变换结果的模
    #
    # pl.plot(f,absY)
    # pl.xlabel('freq(Hz)')
    # pl.title("fft")
    # pl.show()


    if flag:

        pos_Y_from_fft = Y[:Y.size//2]  #10  2
        print(np.argsort(-np.abs(pos_Y_from_fft)))
        print(np.abs(pos_Y_from_fft))
        print(pos_Y_from_fft.size)
        M = pos_Y_from_fft.size
        f = [df*n for n in range(0,M)]

        pl.semilogy(f,np.abs(pos_Y_from_fft))
        pl.xlabel('freq(Hz)')
        pl.title("positiveHalf fft")
        # pl.title("fft in detail")
        pl.show()

    # if flag == 'detail':
        pos_Y_from_fft = Y[:Y.size //10 ]  # 10  2
        print(np.argsort(-np.abs(pos_Y_from_fft)))
        print(np.abs(pos_Y_from_fft))
        print(pos_Y_from_fft.size)

        M = pos_Y_from_fft.size
        f = [df * n for n in range(0, M)]

        # pl.semilogy(f, np.abs(pos_Y_from_fft))
        # pl.xlabel('freq(Hz)')
        # pl.title("fft in detail")
        # pl.show()

    if bandpass == True:   #  只考虑half图 不考虑detail图
        # for order in [3, 6, 9]:
        #     b, a = butter_bandpass(Range[0], Range[1], fs, order=order)
        #     w, h = freqz(b, a)
        #     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        #
        # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Gain')  #  什么是gain
        # plt.grid(True)
        # plt.legend(loc='best')

        # Filter a noisy signal.
        # T = 30
        # totalNbr = int(T * fs)
        # t = np.linspace(0, totalT, totalNbr, endpoint=False)
        a = 0.02
        x = np.load(path)[0:totalNbr]
        x = x[::int((fs//real_freq))]
        t = np.linspace(0, totalT, x.size, endpoint=False)


        plt.figure(2)
        plt.clf()
        plt.plot(t , x , label='Noisy signal')

        y = butter_bandpass_filter(x, Range[0], Range[1], real_freq, order=6)
        plt.plot(t , y , label='Filtered signal (Hz)')
        plt.xlabel('time (seconds)')
        plt.hlines([-a, a], 0, totalT, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc =(0.02,0.7)  )

        plt.show()

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




#if you are sure it is big organ mode, by seeting the flag to true can make the result more convincing!
#if you have given approxmate freq, by setting given to the non zero value, can make the result more convincing!
def test_strategy4windowGlo(thr4str,thr4df,array_3d,df,given):#z = 6#test_strategy4window(array_3d_best12idx_value_freq[:,i,j],df,given)
    # edit mask layer6 of the array_3c


    if given:
        mask1 = np.array((array_3d[4] <= given+thr4df*df), dtype=int)
        mask2 = np.array((array_3d[4] >= given-thr4df*df), dtype=int)
        mask3 = np.array((mask1 == 1) & (mask2 == 1), dtype=int)
    else:
        mask3 = array_3d[5]
    #print(array_3d[4,])#float 64
    #print(array_3d[4,1,1:5])

    # counts = np.bincount(array_3d[4,1])
    # mostcommon = np.argmax(counts)
    # print("the most frequent frequncy value : ",mostcommon)
    mostcommon,times = stats.mode(array_3d[4], axis=None)
    print(mostcommon,times)
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

    array_3d[6] = mask


# processing “mask" again!
    #形态学操作去噪声 去孔洞
    mask_img = mask.astype(np.float32) * 255
    # mask_img = cv2.cvtColor(mask.astype(np.float32) * 255, cv2.COLOR_GRAY2BGR)
    cv2.imshow("before", mask_img)
    cv2.waitKey(0)  #2 4 4   1 1 2
    kernel4E = np.ones((2, 2), np.uint8)#去白点噪声，应该一直很小
    kernel4D = np.ones((4, 4), np.uint8)#thick to original+去黑点（孔洞）  密集的话应该来多轮or增大此kernel  或者这两个参数都略微提高结果最好！
    kernel4E2 = np.ones((4, 4), np.uint8)#before this, make sure u have kicked out all the white/black particles
    mask_img = cv2.erode(mask_img, kernel4E, iterations=1)
    cv2.imshow("Ero", mask_img)
    cv2.waitKey(0)
    mask_img = cv2.dilate(mask_img,kernel4D,iterations=5)#密集的话应该这里来多轮迭代or增大此kernel4d  或者这两个参数都略微提高结果最好！
    cv2.imshow("Dil", mask_img)
    cv2.waitKey(0)
    mask_img = cv2.erode(mask_img, kernel4E2, iterations=2)
    cv2.imshow("final",mask_img)
    cv2.waitKey(0)
    mask = (mask_img/255.0).astype(np.int32)








    #mask = np.uint8(mask)
    array_3d[5] = mask#  change it to mask, the result get improved, as you can see the boundary is not wrongly regarded as info region
    #array([ True, False,  True], dtype=bool)






def visul_savemask(maskDir,mask_path,mask_pathNO,path4array,imgx,imgy,numx,numy):
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

    mask_2dNO = np.load(path4array)[6]
    y_idxNO = np.nonzero(mask_2dNO)[0]
    x_idxNO = np.nonzero(mask_2dNO)[1]
    mask_2dNO = np.zeros((imgy, imgx))
    for y, x in zip(y_idxNO, x_idxNO):
        mask_2dNO[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1

    # mask2d_gray = cv2.cvtColor(mask_2d.astype(np.float32)*255, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(mask_path,mask_2d.astype(np.float32)*255)
    cv2.imwrite(mask_pathNO,mask_2dNO.astype(np.float32) * 255)

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


        cv2.imshow("window", frameYES)

        kk = cv2.waitKey(20) & 0xff  # 实时可视化光流图片，（人自己写好了flow2img函数）
        # Press 'e' to exit the video
        if kk == ord('e'):
            cv2.imwrite(outimgpath,frameYES)
            cv2.imwrite(outimgpathNO,frameNO)

            break








#visulmask("D:\Study\\flownet2-pytorch\infoMat_6height.npy",360,288,20,24)#360 288




def fft_window(givenfreq,thr4str,thr4df,fs, T, path4signal,infoMatPath, real_freq, flag='half', bandpass=False, Range=None, ):  # 默认为half

    totalT = (T[1] - T[0])

    totalNbr = fs * totalT

    # jinjing test
    real_totalNbr = real_freq * totalT

    # 采样步长
    # t = [x/fs for x in range(totalNbr)]

    array_3d = np.load(path4signal)
    num4Timepoints = array_3d.shape[0]
    num4indexY =  array_3d.shape[1]
    num4indexX =  array_3d.shape[2]
    print(array_3d.shape[0], num4indexX, num4indexY)#625 8 8
    array_3d_best12idx_value_freq = np.ones((7,num4indexY,num4indexX))  # index1,2 value1,2 freq 1,2 for most n seconded strong freq+ layer7: noisy mask without too much checking!
    array_3d_best12freqvalue = np.ones((2, num4indexY, num4indexX))
    start = time.time()

    #speed up computation!




    #how to speed here up!
    for i in range(num4indexY):
        for j in range(num4indexX):
            list_2d = array_3d[:,i,j]
            y = list_2d[::int((fs // real_freq))]  # 60/real_freq

            # # 分辨率
            df = real_freq / (real_totalNbr - 1)  # 0.03HZ
            Y = np.fft.fft(y) * 2 / real_totalNbr  # *2/N 反映了FFT变换的结果与实际信号幅值之间的关系

            if flag:
                pos_Y_from_fft = Y[:Y.size // 2]  # 10  2
                # print(np.argsort(-np.abs(pos_Y_from_fft)))
                # print(np.abs(pos_Y_from_fft))
                # print(pos_Y_from_fft.size)
                order_list = np.argsort(-np.abs(pos_Y_from_fft))
#  there is a bug here!
# when the block window is small
                if pos_Y_from_fft.size < 3:
                    print("the number of sampled frames is too less! Please increase <time_range> or <realfreq4samples> in the main!")
                    return False,df

                array_3d_best12idx_value_freq[:,i,j] = [order_list[1],order_list[2],
                                                        np.abs(pos_Y_from_fft)[order_list[1]],
                                                        np.abs(pos_Y_from_fft)[order_list[2]],
                                                        order_list[1]*df,0,0]  #  set mask layer to 0 intially   ?
                #print(array_3d_best12idx_value_freq[:,i,j][5])
                #test_strategy4window(array_3d_best12idx_value_freq[:,i,j],df,given = 1.02)
                #print(array_3d_best12idx_value_freq[:,i,j])
                # array_3d_best12idx_value_freq[0,i,j] = order_list[1]
                # array_3d_best12idx_value_freq[1,i,j] = order_list[2]
                # array_3d_best12idx_value_freq[2,i,j] = np.abs(pos_Y_from_fft)[order_list[1]]
                # array_3d_best12idx_value_freq[3,i,j] = np.abs(pos_Y_from_fft)[order_list[2]]
                # array_3d_best12idx_value_freq[4,i,j] = order_list[1]*df #result
                # array_3d_best12idx_value_freq[5,i,j] = 0 #mask 0/1

                # M = pos_Y_from_fft.size
                # f = [df * n for n in range(0, M)]
                #
                # pl.semilogy(f, np.abs(pos_Y_from_fft))
                # pl.xlabel('freq(Hz)')
                # pl.title("positiveHalf fft")
                # pl.title("fft in detail")
                # pl.show()


    test_strategy4windowGlo(thr4str,thr4df,array_3d_best12idx_value_freq,df,givenfreq)
    end = time.time()
    print('fft time for all windows: ' + str(end - start))
    #print(array_3d_best12idx_value_freq[4,:,:])
    #print(array_3d_best12idx_value_freq[5,:,:])

    np.save(infoMatPath,array_3d_best12idx_value_freq)#
    return True,df







path = 'D:\Study\Datasets\\moreCam\\GFflow\\SIGS_ang.txt.npy'
# path = 'D:\Study\Datasets\\moreCamStable\\GFflow\\SIGS_ang.txt.npy'

# path = 'D:\Study\Datasets\\moreCamBest\\HSflow\\SIGS_mag.txt.npy'
# D:\Study\Datasets\moreCamBestNontrembling\HSflow

path = 'D:\Study\Datasets\\moreCamBestNontrembling\\HSflow\\SIGS_ang.txt.npy'
# path = 'D:\Study\Datasets\\test3L\\GFflow\\SIGS_mag.txt.npy'
# path = 'D:\Study\Datasets\\moreCamBest\\HSflow\\SIGS_ang.txt.npy'



# fft(25,30,path, "detail")
# path = 'D:\Study\Datasets\\moreCam\HSflow\\SIGS_blue.txt.npy'
# fft(30,15,path,10, "detail",True,[1.6,1.9])   #max_freq 5 必须是 25 的因数

#0.25,0.50
#introduce range?
# 通过将第四个参数 从25 调为5  可以看到结果变得更精确和更好

# freq 是真实的帧率   傅里叶变换取样的频率为 real_freq 这是能探测到的频率的上限制
# 应该更小 以达到更精确的结果
#心率的频率也就1.5HZ   呼吸 0.3左右。是用25HZ使得fft分析的频谱过宽，精度就损失了。

# 最好把25HZ降到8HZ也就足够了，也就是最大测到240的心率。这样可以提高4倍精度，
# 心率精度就在1-2之间了。


#fft_window(25,[0,20],"D:\Study\\flownet2-pytorch\SIGS_gray.txt.npy",2.5, "detail",True,[0.1,0.3])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
