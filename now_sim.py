


import pylab as plt
from scipy import stats
from HS_GF_FN_GRAY_func import HS,GF,GRAY,FN
from test import draw2pointsRAW_FFT,on_EVENT_LBUTTONDOWN
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from evaluation import *
from Final_all_functions import pixelwisesig_generate,gridsig_generate


video_list = [6]
givenfreq_list = [1.6]
time_range_list = [[15,20]]
fps = 30
extensionDir = "D:\Study\Datasets\AATEST\\new_short\\"
imgx =   512#360#512#854#720#360#854#360
imgy = 384#288#384#480#288#480#288
fmt = ".avi"#".mp4"#".avi"  #.map4 is ori video suitable for everything except FN,.avi is resize suitable for everything
precision4given = 0.11#
conv_times =1#2#2# 0 #(0,1,2,3)  0 mean no colv   when you set this to true, make sure the note is "ang"
numx_list = [512]#,256,128,64,32]
numy_list = [384]#,192,96,48,24]
mode_list = ["gray"]#,"GF","GF"]#,"FN","FN","HS","HS"]
note_list = [""]#,"mag","ang"]#,"mag","ang","mag","ang"]


def fftwindow2(conv_times, 1, 1, top, top4harmonic, fps, givenfreq,
                                                precision4given,
                                                meansigarrayPath, realfreq4samples, path_prefix):

















#  find 10 s videos to improve quality
for videoname,givenfreq,time_range in zip(video_list,givenfreq_list,time_range_list):
    print(videoname,givenfreq,time_range)
    videoname = str(videoname)
    t = time_range[1] - time_range[0]
    unit = round(t / 5.0)
    top = 5 * unit  # 20#10#20  #keep right there, only take up the highest 20 ID into account
    top = 1*unit
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
            maskset = np.zeros((4, gridnumy, gridnumx), np.bool)

            if gridnumx == 512 and note != "ang":
                pixelwisesig_generate(mode)  # CARRY OUT THIS FOR ONE TIME with pixel wise mose THEN COMMENT OUT
                # pass
            gridsig_generate()  # this is used when increase the grid size,the pixel wise sig to generate obj sig without import video ,OF computing etc.
            flag, df, realtotalNUM = fft_window2(conv_times, 1, 1, top, top4harmonic, fps, givenfreq,
                                                precision4given,
                                                meansigarrayPath, realfreq4samples, path_prefix)  # , [0.1, 0.3])
