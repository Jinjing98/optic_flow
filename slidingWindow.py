from HS_GF_FN_func import HS,GF,GF_window
from signalProcess import fft_window,morph_visul_savemask,playvideowithmask
import cv2
import numpy as np



imgx = 720#360
imgy = 288
gridnumx =360#180#90#360#90#180#360#720#180#72#36#36#18#360#180#18#36#72
gridnumy = 144#36#144#18#36#72#144#288#144#48#24#24#12#288#144#12#24#48

thr4str = 1.1#1.2  bigger stricter   1.0  make no sense
thr4df = 20#1#10  #df~0.05 10is an acceptable decent value for this param 0 means exactly equal,
# since it is float, then get an empty mask. smaller stricter  100 make no sense
#when you are pefectly sure, set this to 1/2

fps = 25
realfreq4samples = 25#5 # mean,"each sec sample 0.5 frame!!"this value should be cautious, the half of it should be bigger than the approomate given freq
String0 = str(gridnumx)+"_"+str(gridnumy)
String = str(gridnumx)+"_"+str(gridnumy)+"_"+str(thr4str)+"_"+str(thr4df)
# String = str(gridnumx)+"_"+str(gridnumy)+"_"+str(thr4str)+"_"+str(thr4df)+"_"+str(realfreq4samples)

time_range =[0,20]#35 [21,31]#[0,20]
givenfreq =6# 0.8#0.35#1.5
videoname = "cardNresp1"#"resp3"#"card1"# \"+videoname+"
#增大k1于增大i1的区别，一个相对大的k1一次迭代就可以消去比k1这个kernel小的噪音，但会导致每一次迭代，有效画面迅速缩小。
# 一个相对小的k1（适用于噪音点也很小的情况）,每次迭代改动一点点，如果有大块噪音，要迭代很多次，一般的，迭代多次会导致图片严重偏离。我们并不喜欢这样。
k1 = int(imgx/gridnumx+2)#must be bigger than the gird, or u just cant delete those noise well!! but should not be too big
k2 = int(imgx*5/4/gridnumx)
k3 = 2#  表示无最终erosion
i1, i2, i3 = 1,4,2#1, 4 ,2#erotion(no white points denoise) dilation(no black points) erosion


#params 4 water in cardNresp1
k1 = int(imgx * 5 / 4 / gridnumx)  # must be bigger than the gird, or u just cant delete those noise well!! but should not be too big
k2 = int(imgx * 5 / 4 / gridnumx + 4)
k3 = 1  # 表示无最终erosion
i1, i2, i3 = 2, 4, 2  # 1, 4 ,2#erotion(no white points denoise) dilation(no black points) erosion



videopath = "D:\Study\Datasets\extension\\"+videoname+"\\"+videoname+".avi"
meansigarrayDIR = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\\"
meansigarrayPath = meansigarrayDIR+"SIGS_gray"+String0+".npy"

infoMatDIR = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\\"
infoMatPath = infoMatDIR+ "infoMat_6height"+String+".npy"
mask_path = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\Mask"+String+".png"  #  the best mask!
mask_img_path = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\mask_img"+String+".png"
mask_pathNO = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\Mask"+String+"NO.png"
mask_img_pathNO = "D:\Study\Datasets\extension\\"+videoname+"\size"+String0+"\mask_img"+String+"NO.png"
maskDir = "D:\Study\Datasets\extension\\"+videoname+"\size"+String+"\\"


#save mean sig array, it gives u the complete data with 25 fps
#GF_window(videopath,meansigarrayPath,time_range,gridnumx,gridnumy)  # 72 grids on x,WE DO NOT NEED TO COMPUTE FOR EACH FRAME WHEN COLLECT ARRAY DATA
# it did fft for sample signals within readfreq, the did fft based on this signal to draw the 6 layer inforMat based on the computed fft spectrum.
#based on sig array, get the raw z(6) array  about the fft result WITH  mask(including test strategy)
flag,df = fft_window(givenfreq,thr4str,thr4df,fps,time_range,meansigarrayPath,infoMatPath,realfreq4samples, "detail",True,[0.1,0.3])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
print("the precision is ",df)
print("the upcoming procedure is ",flag)
if flag:
    # mask2d, mask2dNo is not img, it is 2d numpy array with the same height and width as the image where saved only 0/1, it also save the mask img when calling this func
    # visulise and save the mask# mask2dNO is the mask without the erosion n dilation process
    mask2d, mask2dNO = morph_visul_savemask(maskDir, mask_path, mask_pathNO, infoMatPath, imgx, imgy, gridnumx, gridnumy,k1,k2, k3, i1, i2, i3)
    print(mask2d.shape)
    # https://stackoverflow.com/questions/7587490/converting-numpy-array-to-opencv-array
    # mask2d_gray = cv2.cvtColor(mask2d.astype(np.float32)*255, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite(mask_path,mask2d_gray)
    # visualise video with the mask
    playvideowithmask(fps, time_range, videopath, mask2d, mask2dNO, mask_img_path, mask_img_pathNO)














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





