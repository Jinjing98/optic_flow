from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
# from simple import pixelwisesig_generate
from HS_GF_FN_GRAY_func import HS,GF,GRAY,FN

from test import draw2pointsRAW_FFT, on_EVENT_LBUTTONDOWN_superpixel

def pixelwisesig_generate(mode):

    if mode == "gray":
        GRAY(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )
    if mode == "GF":
        GF(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )
    if mode == "HS":
        HS(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )
    if mode == "FN":
        FN(videopath, meansigarrayDIR,String0, time_range, gridnumx,gridnumy )


def superpixel_array_generate(num4superpixel_input,raw_pixelwise_array_path):
    try:
        sig = np.load(raw_pixelwise_array_path)
    except IOError:
        pixelwisesig_generate(mode)
        sig = np.load(raw_pixelwise_array_path)






    image = sig[0]
    segments = slic(image, n_segments=num4superpixel_input, sigma=5)
    # plt.imshow(segments,cmap="hot")
    # plt.show()

    frames_num = sig.shape[0]
    img_y = sig.shape[1]#384
    img_x = sig.shape[2]#512
    segments_num = np.max(segments)+1
    newsig = np.zeros((frames_num,segments_num))#93

    segments_1D = segments.copy().reshape((img_y*img_x))
    sig_2D = sig.copy().reshape((frames_num,img_x*img_y))


    for seg_idx in range(segments_num):

        print(seg_idx)
        # newsig[:,seg_idx] = np.mean(np.where(segments_1D == seg_idx,sig_2D,0),axis=1)
        newsig[:,seg_idx] = np.mean(np.where(segments == seg_idx,sig,0),axis=(1,2))
        # newsig[:,seg_idx] = np.nanmean(np.where(segments == seg_idx,sig,np.nan),axis=(1,2))

    print(segments_num)
    print(frames_num,sig.shape)
    return newsig,segments


from scipy.signal import find_peaks

import numpy.ma as ma

def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1)) if x.var() !=0 else np.zeros_like(result[n//2 + 1:])
    # acorr = result[:] / (x.var() * np.arange(n,0, -1))

    # lag = np.abs(acorr).argmax() + 1
    # r = acorr[lag-1]
    # if np.abs(r) > 0.5:
    #   print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else:
    #   print('Appears to be not autocorrelated')
    return acorr

def top3_peaks(peaks_int,corr):
    peaks_int_new = np.zeros_like(peaks_int)[:3]
    corr4peaks = corr[peaks_int]
    peaks_order_list = np.argsort(-np.abs(np.absolute(corr4peaks)),axis=0)
    peaks_ID = peaks_order_list[:3]
    print("inter_3ID:",peaks_ID)
    peaks_int_new = peaks_int[peaks_ID]
    print("peaks_int_new:",peaks_int_new)
    return peaks_int_new

def fft_test(new_sig_superpixel,pos_Y_from_fft,df,precision4given,fps,given,topM_ID,final_mask,segments_label_array):
    num4segments = new_sig_superpixel.shape[1]

    ID_expect = given/df
    topM_ID_filter = np.where(topM_ID * df < given + precision4given, topM_ID,
                               0)  # to solve the issue of 1.3/2.6
    topM_ID_filter = np.where(topM_ID_filter * df > given - precision4given, topM_ID_filter,
                               0)  # to solve the issue of 1.3/2.6
    final_mask_ID = np.any(topM_ID_filter, axis=0).astype(bool)  # .astype(np.float64)
    for superpixel_index in range(num4segments):
        final_mask = np.where(segments_label_array == superpixel_index, final_mask_ID[superpixel_index], final_mask)



    #
    # plt.imshow(final_mask, cmap="hot")
    # plt.show()


    return final_mask










def main2(candi_flag,acf_flag):

    raw_pixelwise_array_path = extensionDir+str(videoname)+"\\"+mode_note+"\size512_384\SIGS_"+mode+"_"+timeperiod+note+".npy"
    superpixel_sig_path = extensionDir+str(videoname)+"\\"+mode_note+"\size512_384\sig_superpixel"+str(superpixel_num_esti)+".npy"
    segments_path = extensionDir+str(videoname)+"\\"+mode_note+"\size512_384\segments_label_array"+str(superpixel_num_esti)+".npy"
    #
    #
    #
    # new_sig_superpixel,segments_label_array = superpixel_array_generate(superpixel_num_esti,raw_pixelwise_array_path)
    # np.save(superpixel_sig_path, new_sig_superpixel)
    # np.save(segments_path, segments_label_array)
    #

    try:
        new_sig_superpixel = np.load(superpixel_sig_path)
        segments_label_array = np.load(segments_path)

    except IOError:
        new_sig_superpixel,segments_label_array = superpixel_array_generate(superpixel_num_esti,raw_pixelwise_array_path)
        np.save(superpixel_sig_path, new_sig_superpixel)
        np.save(segments_path, segments_label_array)


    num4superpixel = new_sig_superpixel.shape[1]
    num4frames = new_sig_superpixel.shape[0]
    freq_frame = round(fps/given)
    final_mask = np.ones_like(segments_label_array).astype(bool)
    peaks_int_set = np.zeros([3,num4superpixel]).astype(int)
    len4corr = len(new_sig_superpixel[num4frames//2+1:])
    corr_set = np.zeros([len4corr, num4superpixel])
    pos_Y_from_fft = np.zeros([num4frames // 2, num4superpixel], np.complex128)

    for segment_idx in range(num4superpixel):
        data = new_sig_superpixel[:,segment_idx]
        df = float(fps)/(num4frames-1)
        FFT = np.fft.fft(data) * 2 / num4frames  # *2/N 反映了FFT变换的结果与实际信号幅值之间的关系
        pos_Y_from_fft[:,segment_idx] = FFT[:FFT.size // 2]
    order_list = np.argsort(-np.abs(np.absolute(pos_Y_from_fft)),axis=0)
    topM_ID = order_list[1:(M+1)]


#  beforehand fft candidate filtering
    if candi_flag:
        final_mask = fft_test(new_sig_superpixel,pos_Y_from_fft,df,precision4given,fps,given,topM_ID,final_mask,segments_label_array)

    if acf_flag:
        for superpixel_index in range(num4superpixel):
            data = new_sig_superpixel[:, superpixel_index]

            corr = autocorr(data)
            peaks_int, _ = find_peaks(corr, height=0.3, prominence=0.0, threshold=0.00, width=0)

            # The prominence of a peak measures how much a peak stands out from the surrounding baseline
            # of the signal and is defined as the vertical distance between the peak and its lowest contour line.
            corr_set[:, superpixel_index] = corr
            if len(peaks_int) < 4:
                peaks_int_set[:len(peaks_int), superpixel_index] = peaks_int
            else:
                peaks_int_set[:3, superpixel_index] = top3_peaks(peaks_int, corr)

            # print("acf:", peaks_int_set[:, superpixel_index] - freq_frame)

            if np.min(np.absolute(peaks_int_set[:, superpixel_index] - freq_frame)) > 3:
                final_mask = np.where(segments_label_array == superpixel_index, 0, final_mask)









    plt.imshow(final_mask,cmap="hot")
    plt.show()
    final_mask = final_mask.copy().astype(np.float32) * 255

    cv2.namedWindow("image")
    # img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP = (
    # mask_2dCP, gridwidth, gridheight, pos_Y_from_fft, visuliseFFT, visuliseRAW, df, path4signal, array_3d,
    # maskNfreqID_infoMat[1])

    params = (corr_set,peaks_int_set,new_sig_superpixel,segments_label_array,final_mask,freq_frame)
    # this func can interatively show curve
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN_superpixel,params)
    cv2.imshow("image", final_mask)
    cv2.waitKey(0)  # non vis interation!



superpixel_num_esti = 500   #100 500 2500
M = 3
precision4given = 0.11
# raw_pixelwise_array_path = "D:\Study\Datasets\AATEST\instru_pulse\\10\gray\size512_384\SIGS_gray_0_5.npy"
# superpixel_sig_path = "D:\Study\Datasets\AATEST\instru_pulse\\10\gray\size512_384\sig_superpixel"+str(superpixel_num_esti)+".npy"
# segments_path = "D:\Study\Datasets\AATEST\instru_pulse\\10\gray\size512_384\segments_label_array"+str(superpixel_num_esti)+".npy"

    # D:\Study\Datasets\AATEST\instru_pulse\6\gray\size512_384
videoname = 9#
candi_flag = 0#False
acf_flag = 1#False
videoname = str(videoname)
fmt = ".avi"
given = 1.2#1.6
fps = 30
time_range = [0,5]#[15,20]
extensionDir = "D:\Study\Datasets\AATEST\\instru_pulse\\"
mode = "FN"#"GF"
note = "ang"#"ang"
mode_note = mode+"\\"+note

meansigarrayDIR = extensionDir + str(videoname) + "\\"
timeperiod = str(time_range[0])+"_"+str(time_range[1])
String0 = str(512) + "_" + str(384)
gridnumx = 512
gridnumy = 384
videopath = extensionDir + videoname + fmt
main2(candi_flag,acf_flag)


















# #
# # img = img_as_float(astronaut()[::2, ::2])
# # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# #
# # print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))
# # plt.imshow(mark_boundaries(img, segments_quick))
# #
# # plt.show()
#
#
# # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# # args = vars(ap.parse_args())
# # # load the image and convert it to a floating point data type
# # image = img_as_float(io.imread(args["image"]))
# image = img_as_float(io.imread("D:\Study\Datasets\surgery_test\surgery_test_result\\150.png"))
# image = cv2.imread("D:\Study\Datasets\surgery_test\surgery_test_result\\150.png")
# # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# for numSegments in (100,):
# 	# apply SLIC and extract (approximately) the supplied number
# 	# of segments
# 	segments = slic(image, n_segments = numSegments, sigma = 5)
# 	# show the output of SLIC
# 	# fig = plt.figure("Superpixels -- %d segments" % (numSegments))
# 	# ax = fig.add_subplot(1, 1, 1)
# 	# ax.imshow(mark_boundaries(image, segments))
# 	# plt.axis("off")
# plt.imshow(segments,cmap="hot")
# # show the plots
# plt.show()
#
#
#
#
#
#
#
# def GRAY(videopath,sigDIR,String0,time_range,gridnumX,gridnumY ):  # choose ranges given a video length to collect npy
#     #meansigarrayPath = meansigarrayDIR+mode+"\size"+String0+"\\"+"SIGS_"+mode+"_"+String0+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"
#     sigpath = sigDIR+"gray"+"\size"+String0+"\\"+"SIGS_gray"+"_"+str(time_range[0])+"_"+str(time_range[1])+".npy"
#
#
#     cap = cv2.VideoCapture(videopath)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     _, frame1 = cap.read()
#     prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#
#     lower = fps * time_range[0]
#     upper = fps * time_range[1]
#     framenum = upper - lower
#
#
#     gridWidth = int(width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
#     gridHeight = int(height / gridnumY)  # 240
#     SIGS_gray = np.zeros((framenum, gridnumY, gridnumX))  # 4,3
#
#
#     i = 1
#     timepoint = 0
#     while (i):
#
#         ret, frame2 = cap.read()
#         if i < lower:  # 直接从好帧开始运行
#             i += 1
#             continue
#         if i >= upper:
#             break
#         if ret != True:
#             break
#         next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
#
#         # before it took around 3 mins, now it is way faster based on numpy vectorization
#         # https://codingdict.com/questions/179693
#         next = next.reshape(gridnumY, 1, gridnumX, 1)
#
#         # print(next[0,:,0,:])
#
#         SIGS_gray[timepoint] = next.mean(axis=(1, 3))
#
#         timepoint += 1
#         i += 1
#
#     # np.save(sigpath+"GFSIGS_ang.txt",SIGS_ang)
#     # np.save(sigpath+"GFSIGS_mag.txt",SIGS_mag)
#     np.save(sigpath, SIGS_gray)
#     # np.save("D:\\Study\\Datasets\\AEXTENSION\\Cho80_extension\\static_cam\\pulseNstatic\\1\\gray\\size854_480\\SIGS_gray854_480.npy",SIGS_gray)
#
#
#
#
#
# def gridsig_generate():
#     path2sig = meansigarrayDIR + mode + "\\" + note + "\size" + str(imgx) + "_" + str(
#         imgy) + "\\" +"SIGS_"+mode+"_"+str(time_range[0])+"_"+str(time_range[1])+note+".npy"
#     newpath2sig = meansigarrayDIR + mode + "\\" + note + "\size" + str(gridnumx) + "_" + str(
#         gridnumy) + "\\" + "SIGS_"+mode+"_"+str(time_range[0])+"_"+str(time_range[1])+note+".npy"
#
#     try:
#         sig = np.load(path2sig)
#     except IOError:
#         # pixelwisesig_generate(mode)
#         sig = np.load(path2sig)
#     sig = sig.reshape(-1, gridnumy, int(imgy / gridnumy), gridnumx, int(imgx / gridnumx))
#     newsig = sig.mean(axis=(2, 4))
#     np.save(newpath2sig, newsig)




