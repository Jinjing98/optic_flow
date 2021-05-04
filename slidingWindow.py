from HS_GF_FN_func import HS,GF,GF_window
from signalProcess import fft_window,visulmask,playvideowithmask
import cv2
import numpy as np
from PIL import Image

videopath = "D:\Study\Datasets\extension\card1\card1.avi"""

meansigarrayDIR = "D:\Study\Datasets\extension\card1\size72_48\\"
meansigarrayPath = meansigarrayDIR+"SIGS_gray_72_48.npy"

infoMatDIR = "D:\Study\Datasets\extension\card1\size72_48\\"
infoMatPath = infoMatDIR+ "infoMat_6height_72_48.npy"

mask_path = "D:\Study\Datasets\extension\card1\size72_48\Mask72_48.png"
mask_img_path = "D:\Study\Datasets\extension\card1\size72_48\mask_img72_48.png"




#save mean sig array
GF_window(videopath,meansigarrayPath,[0,20],144,96)  # 72 grids on x,WE DO NOT NEED TO COMPUTE FOR EACH FRAME WHEN COLLECT ARRAY DATA
#based on sig array, get the raw z(6) array  about the fft result WITH  mask(including test strategy)
df = fft_window(25,[0,20],meansigarrayPath,infoMatPath,2.5, "detail",True,[0.1,0.3])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
# visulise the mask
mask2d = visulmask(infoMatPath,360,288,144,96)
print(mask2d.shape)
#https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array
mask2d_gray = cv2.cvtColor(mask2d.astype(np.float32)*255, cv2.COLOR_GRAY2BGR)
cv2.imwrite(mask_path,mask2d_gray)
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





