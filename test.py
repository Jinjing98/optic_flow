import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#ComplexWarning: Casting complex values to real discards the imaginary part
# mask2D = np.any(top5_ID, axis=0)  # .astype(np.float64)
# delta_top5_ID_5D = top5_ID - ID_expect
# posi_neg_mask = np.where(delta_top5_ID_5D < 0, 0, 1)  # freq<given is 0
# delta_top5_ID_1D = np.min(np.abs(delta_top5_ID_5D), axis=0)
# Best_ID_1D = delta_top5_ID_1D +

#这里导入你自己的数据
#......
#......
# gridwidth = int(imgx / numx)
# gridheight = int(imgy / numy)
# mask_2d = np.load(path4array)[5]
# print(mask_2d.shape)  # 12 18
# print(mask_2d)
# # print(np.nonzero(mask_2d))
# # print(np.nonzero(mask_2d)[0])#189
# y_idx = np.nonzero(mask_2d)[0]
# x_idx = np.nonzero(mask_2d)[1]
# mask_2d = np.zeros((imgy, imgx))

import matplotlib.pyplot as plt
import numpy as np


# fps = 25
# realfreq4samples = 25
#
# t = 10
# df = 1/t
# real_totalNbr = 25/(int(25/realfreq4samples))*t
# vispos_YX = [1,7]  #  non pulse
# vispos_YX2 = [240,425]
# videoname = "BinW25_5hz"
# mode = "gray"#"gray"  #gray   HS  FN GF
# note = ""#"mag"#"mag"# "ang" "mag" ""
# time_range =[0,10]
# imgx = 854#720#360#854#360
# imgy =  480#288#480#288
# gridnumx =   854#512#256#128#32#64#128#256#512#32#64#128#256#512#45#90#180#360#180#360#512#854#720#360#854#360#180#90#360#90#180#360#720#180#72#36#36#18#360#180#18#36#72
# gridnumy =    480#384#192#96#24#48#96#192#384#24#48#96#192#384#36#72#144#288#144#288#384#480#288#480#288#144#36#144#18#36#72#144#288#144#48#24#24#12#288#144#12#24#48
# extensionDir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\\arti_videos\\"#"D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\"
# meansigarrayDIR = extensionDir+videoname+"\\"


def draw2pointsRAW_FFT(vispos_YX,vispos_YX2,realfreq4samples,t,mode,note,gridnumx,gridnumy,time_range,meansigarrayDIR,df,videoname):
    real_totalNbr = 25 / (int(25 /realfreq4samples)) * t
    path2sig = meansigarrayDIR + mode + "\\" + note + "\size" + str(gridnumx) + "_" + str(
        gridnumy) + "\\" + "SIGS_" + mode + "_" + str(gridnumx) + "_" + str(gridnumy) + "_" + str(
        time_range[0]) + "_" + str(time_range[1]) + note + ".npy"
    array3d = np.load(path2sig)  # end samplerate

    Y = vispos_YX[0]
    X = vispos_YX[1]
    Y2 = vispos_YX2[0]
    X2 = vispos_YX2[1]

    sig = array3d[:, Y, X]
    fftsig = np.abs(np.absolute(np.fft.fft(sig)) * 2 / real_totalNbr)  # *2/N 反映了FFT变换的结果与实际信号幅值之间的关系
    fftsighalf = fftsig[:fftsig.size//2]

    sig2 = array3d[:, Y2, X2]
    fftsig2 = np.abs(np.absolute(np.fft.fft((sig2 )) )* 2 / real_totalNbr)  # *2/N 反映了FFT变换的结果与实际信号幅值之间的关系
    fftsighalf2 = fftsig2[:fftsig2.size//2]


    M = fftsighalf.size
    f = [df * n for n in range(0, M)]

    # pl.semilogy(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))#?
    plt.plot(f, fftsighalf,color='red', label="Y,X " + str(Y) + "_" + str(X))
    plt.plot(f, fftsighalf2, color='gray', label="Y,X "+str(Y2)+"_"+str(X2))
    plt.xlabel('freq(Hz)')
    plt.title("positiveHalf fft video:" +videoname+" "+mode+" "+note)
    sub_axix = filter(lambda x: x % 200 == 0, f)
    plt.legend()  # 显示图例
    plt.savefig(path2sig[:-4]+"FFT"+ str(Y)+"_"+str(X)+"--"+ str(Y2)+"_"+str(X2)+".png")
    plt.show()








    #Y  X
    train1 = array3d[:,Y,X][:100]
    train2 = array3d[:,Y2,X2][:100]
    # train3 = array3d[:,45,21][:90]
    # train4 = array3d[:,30,77][:90]

    # train5 = array3d[:,42,21][:150]
    # train6 = array3d[:,43,21][:150]
    # train7 = array3d[:,48,21][:90]
    # train8 = array3d[:,275,356][:90]
    # print(2)

    #开始画图

    x_axix = [i for i in range(100)]#
    sub_axix = filter(lambda x:x%200 == 0, x_axix)
    plt.title('SRC signal tendency and magnitude video:'+videoname+" "+mode+" "+note+" "+"realfreq4samples"+str(realfreq4samples))
    plt.plot(x_axix, train1, color='red', label="Y,X "+str(Y)+"_"+str(X))
    plt.plot(x_axix, train2 , color='gray', label="Y,X "+str(Y2)+"_"+str(X2))

    plt.legend() # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.savefig(path2sig[:-4] + "RAW" + str(Y)+"_"+str(X)+"--"+ str(Y2)+"_"+str(X2)+".png")
    plt.show()


import cv2
import numpy as np

# # Picture path
# img = cv2.imread('D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\3\gray\size32_24\\Mask32_24_3.74_20_2_5_1.4.png')
# gridwidth = 16
# gridheight = 16

def on_EVENT_LBUTTONDOWN(event, x, y,flag,img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP):
    if event == cv2.EVENT_LBUTTONDOWN:
        vispos_YX = []
        vispos_YX.append(int(np.ceil(y/img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[2]))-1)  #-1  since, the mouse auto recoginize frame from (1,1)
        vispos_YX.append(int(np.ceil(x/img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[1]))-1)
        yx = "(%d,%d)" % (vispos_YX[0],vispos_YX[1])
        cv2.circle(img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[0], (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[0], yx, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,255), thickness=1)
        cv2.imshow("image", img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[0])
        Y = vispos_YX[0]
        X = vispos_YX[1]
        print("Y: ",Y," X: ", X )
        
        if img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[4]:  # is visFFT

            M = img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[3][:, Y, X].size
            df = img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[6]
            f = [df * n for n in range(0, M)]
            magnitudelist = np.abs(np.absolute(img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[3][:, Y, X]))
            # pl.semilogy(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))#?
            plt.plot(f, magnitudelist)
            plt.xlabel('freq(Hz)')
            plt.title("positiveHalf fft YX" + str(Y) + "_" + str(X))
            # plt.savefig(img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[7][:-4] + "_" + str(Y) + "_" + str(X) + ".png")
            plt.show()
            freqIDMAP = img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[-1]
            print(" BestFreq: ",freqIDMAP[Y][X]*df," mag4BestFreq: ",magnitudelist[freqIDMAP[Y][X]])
        if img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[5]:  # is visRAW

            train1 = img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[8][:, Y, X][:100]
            x_axix = [i for i in range(100)]

            plt.plot(x_axix, train1, color='red', label="Y,X " + str(Y) + "_" + str(X))
            # plt.plot(x_axix, train2, color='gray', label="Y,X " + str(Y2) + "_" + str(X2))

            # plt.legend()  # 显示图例


            # pl.semilogy(f, np.abs(np.absolute(pos_Y_from_fft[:,Y,X])))#?
            plt.xlabel('frame id')
            plt.title("RAW SIGNAL AT GRID POS YX" + str(Y) + "_" + str(X))
            # plt.savefig(img_width_height_fftNUMPY_VISFFTflag_VISRAWflag_df_path4curve_arrayRAW_freqIDMAP[7][:-4] + "_" + str(Y) + "_" + str(X) + "RAW.png")
            plt.show()





# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN,(img,gridwidth,gridheight,pos_Y_from_fft))
# cv2.imshow("image", img)
# cv2.waitKey(0)
