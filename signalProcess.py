import numpy as np
import pylab as pl
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import butter, lfilter


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


def fft(fs,totalT,path, real_freq,flag = 'half',bandpass = False,Range = None):   #  默认为half




    #jinjing


    # 采样频率
    # fs=25   #  反映在我们的应用中是视频帧率
    # totalT = 25 # 单位为s 30 25
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
    y = y[::(fs//real_freq)]
    # y = np.load(path)[24*30:24*30+totalNbr]   #debug
    t = [totalT*x/y.size for x in range(y.size)]


    # print(y)
    # print(y.size)
    # y = [ 3.0 * np.cos(2.0 *np.pi * 0.50 * t0  )
    #      # + 1.5 * np.cos(800* t0 + np.pi * 90/180)
    #      # +  1.0 * np.cos(2.0 * np.pi * 800 * t0 + np.pi * 120/180)
    #      # +  2.0 * np.cos(1/2.0 /np.pi * 220 * t0 + np.pi * 30/180)
    #      for t0 in t ]




    pl.plot(t,y)   #   y is orignal signal
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

        pl.plot(f,np.abs(pos_Y_from_fft))
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

        pl.plot(f, np.abs(pos_Y_from_fft))
        pl.xlabel('freq(Hz)')
        # pl.title("positiveHalf fft")
        pl.title("fft in detail")
        pl.show()

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
        x = x[::(fs//real_freq)]
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
        plt.legend(loc='upper left')

        plt.show()












path = 'D:\Study\Datasets\\moreCam\\GFflow\\SIGS_ang.txt.npy'
# path = 'D:\Study\Datasets\\moreCamStable\\GFflow\\SIGS_ang.txt.npy'

path = 'D:\Study\Datasets\\moreCamBest\\HSflow\\SIGS_ang.txt.npy'
# D:\Study\Datasets\moreCamBestNontrembling\HSflow

path = 'D:\Study\Datasets\\moreCamBestNontrembling\\HSflow\\SIGS_graybetter.txt.npy'


# fft(25,30,path, "detail")
# path = 'D:\Study\Datasets\\moreCam\HSflow\\SIGS_blue.txt.npy'
fft(25,16,path,25, "detail",True,[0.25,0.45])   #max_freq 5 必须是 25 的因数


# 通过将第四个参数 从25 调为5  可以看到结果变得更精确和更好

# freq 是真实的帧率   傅里叶变换取样的频率为 real_freq 这是能探测到的频率的上限制
# 应该更小 以达到更精确的结果
#心率的频率也就1.5HZ左右。是用25HZ使得fft分析的频谱过宽，精度就损失了。
# 最好把25HZ降到8HZ也就足够了，也就是最大测到240的心率。这样可以提高4倍精度，
# 心率精度就在1-2之间了。


