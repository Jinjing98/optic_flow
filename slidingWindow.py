from HS_GF_FN_func import HS,GF,GF_window
from signalProcess import fft_window,visul_savemask,playvideowithmask
import cv2
import numpy as np



imgx = 360#720#360
imgy = 288
gridnumx = 36#36#18#360#180#18#36#72
gridnumy = 24#24#12#288#144#12#24#48
thr4str = 1.1#1.2  bigger stricter   1.0  make no sense
thr4df = 1#10  #df~0.05 10is an acceptable decent value for this param 0 means exactly equal,
# since it is float, then get an empty mask. smaller stricter  100 make no sense
#when you are pefectly sure, set this to 1/2
String0 = str(gridnumx)+"_"+str(gridnumy)
String = str(gridnumx)+"_"+str(gridnumy)+"_"+str(thr4str)+"_"+str(thr4df)
fps = 25
realfreq4samples = 5#1#5 # this value should be cautious, the half of it should be bigger than the approomate given freq
time_range = [0,20]#[0,15]#[0,20]
givenfreq = 1.5#0.4#1.5
videoname = "card1"#"resp1"#"card1"# \"+videoname+"



videopath = "D:\Study\Datasets\extension\\"+videoname+"\\"+videoname+".avi"

meansigarrayDIR = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\\"
meansigarrayPath = meansigarrayDIR+"SIGS_gray"+String0+".npy"

infoMatDIR = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\\"
infoMatPath = infoMatDIR+ "infoMat_6height"+String+".npy"

mask_path = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\Mask"+String+".png"  #  the best mask!
mask_img_path = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\mask_img"+String+".png"

maskDir = "D:\Study\Datasets\extension\\"+videoname+"\size"+String+"\\"















#save mean sig array
#GF_window(videopath,meansigarrayPath,time_range,gridnumx,gridnumy)  # 72 grids on x,WE DO NOT NEED TO COMPUTE FOR EACH FRAME WHEN COLLECT ARRAY DATA
#based on sig array, get the raw z(6) array  about the fft result WITH  mask(including test strategy)
df = fft_window(givenfreq,thr4str,thr4df,fps,time_range,meansigarrayPath,infoMatPath,realfreq4samples, "detail",True,[0.1,0.3])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
#test stage for the 5th layer of infoMat, set 0/1 for the 5th layer


# visulise the mask
mask2d = visul_savemask(maskDir,mask_path,infoMatPath,imgx,imgy,gridnumx,gridnumy)
print(mask2d.shape)
#https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array
# mask2d_gray = cv2.cvtColor(mask2d.astype(np.float32)*255, cv2.COLOR_GRAY2BGR)
# cv2.imwrite(mask_path,mask2d_gray)
#visualise video with the mask
playvideowithmask(videopath,mask2d,mask_img_path)











#def fft(fs,T,path, real_freq,flag = 'half',bandpass = False,Range = None,):

    # path = 'D:\Study\Datasets\\test3L\\GFflow\\SIGS_mag.txt.npy'
    # path = 'D:\Study\Datasets\\moreCamBest\\HSflow\\SIGS_ang.txt.npy'

    # fft(25,30,path, "detail")
    # path = 'D:\Study\Datasets\\moreCam\HSflow\\SIGS_blue.txt.npy'
    # fft(30,15,path,10, "detail",True,[1.6,1.9])   #max_freq 5 必须是 25 的因数

    # 0.25,0.50
    # introduce range?
    # 通过将第四个参数 从25 调为5  可以看到结果变得更精确和更好

    # freq 是真实的帧率   傅里叶变换取样的频率为 real_freq 这是能探测到的频率的上限制
    # 应该更小 以达到更精确的结果
    # 心率的频率也就1.5HZ   呼吸 0.3左右。是用25HZ使得fft分析的频谱过宽，精度就损失了。

    # 最好把25HZ降到8HZ也就足够了，也就是最大测到240的心率。这样可以提高4倍精度，
    # 心率精度就在1-2之间了。





