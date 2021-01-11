import numpy as np
import pylab as pl
import math


#考虑滑动均值滤波和带通滤波进一步优化  可视化频率   （已知区间  PCA  可信度衡量 统计学

def fft(fs,totalT,path,flag = 'half'):   #  默认为half




    #jinjing


    # 采样频率
    # fs=25   #  反映在我们的应用中是视频帧率
    # totalT = 25 # 单位为s 30 25
    totalNbr = fs * totalT
    # 采样步长
    t = [x/fs for x in range(totalNbr)]
    """
    设计的采样值
    假设信号y由4个周期信号叠加所得,如下所示
    """
    # y = np.load('D:\Study\Datasets\\Cam\GFflow\\SIGS_mag.txt.npy')[0:totalNbr]
    # y = np.load('D:\Study\Datasets\\Cam\HSflow\\SIGS_ang.txt.npy')[0:totalNbr]
    # y = np.load('D:\Study\Datasets\\Cam\\NNflow\\SIGS_ang.txt.npy')[0:totalNbr]
    y = np.load(path)[0:totalNbr]

    # print(y)
    # print(y.size)
    # y = [ 3.0 * np.cos(2.0 *np.pi * 0.50 * t0  )
    #      # + 1.5 * np.cos(800* t0 + np.pi * 90/180)
    #      # +  1.0 * np.cos(2.0 * np.pi * 800 * t0 + np.pi * 120/180)
    #      # +  2.0 * np.cos(1/2.0 /np.pi * 220 * t0 + np.pi * 30/180)
    #      for t0 in t ]




    pl.plot(t,y)
    pl.xlabel('time(s)')
    pl.title("original signal")
    pl.show()




    """
    现在对上述信号y在0-1秒时间内进行频谱分析，
    
    本案例中采样频率为1048Hz,即单位时间内采样点数为1048
    """
    # # 采样点数
    # N=len(t) ie totalNbr
    # # 采样频率
    # fs=25.0
    # # 分辨率
    df = fs/(totalNbr-1)   #0.03HZ
    # # 构建频率数组
    f = [df*n for n in range(0,totalNbr)]   #  最大频率为30HZ
    Y = np.fft.fft(y)*2/totalNbr  #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
    # absY = [np.abs(x) for x in Y]      #求傅里叶变换结果的模
    #
    # pl.plot(f,absY)
    # pl.xlabel('freq(Hz)')
    # pl.title("fft")
    # pl.show()


    if flag == 'half':

        pos_Y_from_fft = Y[:Y.size//2]  #10  2
        M = pos_Y_from_fft.size
        f = [df*n for n in range(0,M)]

        pl.plot(f,np.abs(pos_Y_from_fft))
        pl.xlabel('freq(Hz)')
        pl.title("positiveHalf fft")
        # pl.title("fft in detail")
        pl.show()

    if flag == 'detail':
        pos_Y_from_fft = Y[:Y.size //10 ]  # 10  2
        M = pos_Y_from_fft.size
        f = [df * n for n in range(0, M)]

        pl.plot(f, np.abs(pos_Y_from_fft))
        pl.xlabel('freq(Hz)')
        # pl.title("positiveHalf fft")
        pl.title("fft in detail")
        pl.show()



path = 'D:\Study\Datasets\\moreCam\HSflow\\SIGS_gray.txt.npy'
# fft(25,30,path, detail)
path = 'D:\Study\Datasets\\moreCam\HSflow\\SIGS_blue.txt.npy'
# fft(25,30,path, detail)