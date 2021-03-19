import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


path =  "D:\Study\Datasets\pairs\\video64\\diff150NN.png"
pathHS =  "D:\Study\Datasets\pairs\\video64\\diff150HS.png"
pathGF =  "D:\Study\Datasets\pairs\\video64\\diff150GF.png"
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
imgHS = cv2.imread(pathHS,cv2.IMREAD_GRAYSCALE)
imgGF = cv2.imread(pathGF,cv2.IMREAD_GRAYSCALE)

# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()
x = np.arange(256)


path =  "D:\Study\Datasets\hamlyn\FNerror\\"
pathHS =  "D:\Study\Datasets\hamlyn\HSerror\\"
pathGF =  "D:\Study\Datasets\hamlyn\GFerror\\"
initpath = path+'capture-20110222T141957Z_1.png'
initpathHS = pathHS+'capture-20110222T141957Z_1.png'
initpathGF = pathGF+'capture-20110222T141957Z_1.png'


# #just need to edit line 57/58  and below lines
# path =  "D:\Study\Datasets\cholec80\FNerror\\"
# pathHS =  "D:\Study\Datasets\cholec80\HSerror\\"
# pathGF =  "D:\Study\Datasets\cholec80\GFerror\\"
# initpath = path+'video01_1.png'
# initpathHS = pathHS+'video01_1.png'
# initpathGF = pathGF+'video01_1.png'




num = 1
img = cv2.imread(initpath,cv2.IMREAD_GRAYSCALE).flatten()
imgHS = cv2.imread(initpathHS,cv2.IMREAD_GRAYSCALE).flatten()
imgGF = cv2.imread(initpathGF,cv2.IMREAD_GRAYSCALE).flatten()
for f_name in os.listdir(path):
    next_img = cv2.imread(path+f_name,cv2.IMREAD_GRAYSCALE).flatten()
    img = np.hstack((img,next_img))
    num += 1
for f_name in os.listdir(pathHS):
    next_img = cv2.imread(pathHS+f_name,cv2.IMREAD_GRAYSCALE).flatten()
    imgHS = np.hstack((imgHS,next_img))
    # num += 1
for f_name in os.listdir(pathGF):
    next_img = cv2.imread(pathGF+f_name,cv2.IMREAD_GRAYSCALE).flatten()
    imgGF = np.hstack((imgGF,next_img))
    # num += 1
x = np.arange(256)


totalNum = 512*256*(num-35)+256*256*6*5+256*192*1*5
# totalNum = 768*384*num

































hist = cv2.calcHist([img], [0], None, [256], [0, 256])    #   每16个像素当作一个单元  共256个条条  选公因数
hist = hist.flatten()
print(np.sum(hist))
histHS = cv2.calcHist([imgHS], [0], None, [256], [0, 256])
histHS = histHS.flatten()
histGF = cv2.calcHist([imgGF], [0], None, [256], [0, 256])
histGF = histGF.flatten()
cum_hist = np.cumsum(hist)
cum_histHS = np.cumsum(histHS)
cum_histGF = np.cumsum(histGF)





# plt.plot(x,hist/totalNum,alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
# plt.plot(x,histHS/totalNum,alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
# plt.plot(x,histGF/totalNum,alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
# plt.subplot(2,1,1)
plt.plot(x[0:30],(hist/totalNum)[0:30],alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
plt.plot(x[0:30],(histHS/totalNum)[0:30],alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
plt.plot(x[0:30],(histGF/totalNum)[0:30],alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
plt.legend(loc='upper left', bbox_to_anchor=(0.85, 0.55))
plt.ylabel('probability')
plt.xlabel('error')
plt.title('\n'+"PMF of error range in [0,30]")
plt.show()

# plt.subplot(2,1,2)
plt.plot(x[0:30],(cum_hist/totalNum)[0:30],alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
plt.plot(x[0:30],(cum_histHS/totalNum)[0:30],alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
plt.plot(x[0:30],(cum_histGF/totalNum)[0:30],alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
# plt.scatter(x,hist/totalNum,s = 4, c = 'r',marker = "o",linewidths = 0.1,label= 'NN')
# plt.scatter(x,histHS/totalNum,s = 4, c = 'g',marker = "x",linewidths = 0.1,label= 'HS')
# plt.plot(x, y1, color="r", linestyle="-", marker="^", linewidth=1)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.55))
plt.ylabel('probability')
plt.xlabel('error')
plt.title( '\n'+"CDF of error range in [0,30]")
plt.show()


#cdf


plt.plot(x,hist/totalNum,alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
plt.plot(x,histHS/totalNum,alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
plt.plot(x,histGF/totalNum,alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))
plt.ylabel('probability')
plt.xlabel('error')
plt.title( '\n'+"PMF of error  ")
plt.show()
print(max(hist/totalNum),min(hist/totalNum))
print(hist/totalNum)

# plt.step(x,cum_hist/totalNum,color="b",lw=1)
# plt.title("cdf of error")
# plt.grid()
# plt.show()

plt.plot(x,cum_hist/totalNum,alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
plt.plot(x,cum_histHS/totalNum,alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
plt.plot(x,cum_histGF/totalNum,alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))
plt.ylabel('probability')
plt.xlabel('error')
plt.title( '\n'+"CDF of error  ")
plt.show()










# PMF of number of each pixel error

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.rcParams['font.size'] = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.bar(x,cum_hist/totalNum,zs=1, zdir='y', color='r', alpha=1 ,label = 'NN')
ax.bar(x,cum_histHS/totalNum,zs=2, zdir='y', color='g', alpha=1 ,label = 'HS')
ax.bar(x,cum_histGF/totalNum,zs=3, zdir='y', color='b', alpha=1, label = 'GF' )
ax.grid(False)
# fig.xticks([])
# plt.yticks([])
# ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(x))
ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(hist/totalNum))
ax.set_xlabel('Error')
ax.set_zlabel('Probability(PMF)')
ax.set_ylabel('Methods')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.99))
plt.title( '\n'+"CDF of error ")

plt.show()



#boxplot OF PIXEL VALUES
import matplotlib.pyplot as plt

# 读取数据

# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# plt.title(' boxplot OF PIXEL VALUES', fontsize=20)  # 标题，并设定字号大小
# labels = 'NN', 'GF','HS' # 图例

# vert=False:水平箱线图；showmeans=True：显示均值
# plt.boxplot([img.ravel(),imgGF.ravel(),imgHS.ravel()], labels=labels, vert=False, showmeans=True)
# plt.show()  # 显示图像
# print(hist,hist/totalNum)

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x3 = np.concatenate((np.arange(256) ,np.arange(256),np.arange(256)))
y3 = np.concatenate([np.arange(3) for i in range(256)])
z3 = np.concatenate([np.zeros(256)for i in range(3)])

dx = np.ones(1)
dy = np.ones(1)
dz = [1,2,3,4,5,6,7,8,9,10]

ax1.bar3d(x3, y3, z3, dx, dy,np.concatenate([cum_hist,cum_histHS,cum_histGF]))


ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()












# single cdf

cum_hist = np.cumsum(hist)
plt.plot(x,cum_hist/totalNum)
plt.title("cdf of error")
plt.show()
plt.step(x,cum_hist/totalNum,color="b",lw=1)
plt.title("cdf of error")
plt.grid()
plt.show()
print(max(cum_hist/totalNum),min(cum_hist/totalNum))
print(cum_hist/totalNum)
meanError = np.mean(imgHS)
# 这两个指标定义error出现的数量
preciseRateStrict = cum_histHS[0]/totalNum     #define a error when it is not zero   strict error
preciseRateRough = cum_histHS[25]/totalNum     #define error when the error is bigger than 16 pixels

# 这几个指标定义error 偏离的程度  （与error值有关）
meanError = np.mean(imgHS)
strictErrorMean = np.sum(histHS[1:-1] * x[1:-1])/(cum_histHS[-1] - cum_histHS[0])   #per pixel error value range in 1-255
roughErrorMean = np.sum(histHS[26:-1] * x[26:-1])/(cum_histHS[-1] - cum_histHS[25])   #per pixel error value range in 26-255
var = np.var(imgHS, ddof=1)


print("HS:\n  preciseRateStrict:", preciseRateStrict,  "\n mean error for strict errors ", strictErrorMean, "\npreciserateRough : ",preciseRateRough,
          "\n mean error for rough errors ", roughErrorMean, "\n global mean error:", meanError,'\n varience:', var)


meanError = np.mean(imgGF)
# 这两个指标定义error出现的数量
preciseRateStrict = cum_histGF[0]/totalNum     #define a error when it is not zero   strict error
preciseRateRough = cum_histGF[25]/totalNum     #define error when the error is bigger than 16 pixels

# 这几个指标定义error 偏离的程度  （与error值有关）
meanError = np.mean(imgGF)
strictErrorMean = np.sum(histGF[1:-1] * x[1:-1])/(cum_histGF[-1] - cum_histGF[0])   #per pixel error value range in 1-255
roughErrorMean = np.sum(histGF[26:-1] * x[26:-1])/(cum_histGF[-1] - cum_histGF[25])   #per pixel error value range in 26-255
var = np.var(imgGF, ddof=1)

print("GF:\n  preciseRateStrict:", preciseRateStrict,  "\n mean error for strict errors ", strictErrorMean, "\npreciserateRough : ",preciseRateRough,
          "\n mean error for rough errors ", roughErrorMean, "\n global mean error:", meanError,'\n varience:', var)


meanError = np.mean(img)
# 这两个指标定义error出现的数量
preciseRateStrict = cum_hist[0]/totalNum     #define a error when it is not zero   strict error
preciseRateRough = cum_hist[25]/totalNum     #define error when the error is bigger than 16 pixels

# 这几个指标定义error 偏离的程度  （与error值有关）
meanError = np.mean(img)
strictErrorMean = np.sum(hist[1:-1] * x[1:-1])/(cum_hist[-1] - cum_hist[0])   #per pixel error value range in 1-255
roughErrorMean = np.sum(hist[26:-1] * x[26:-1])/(cum_hist[-1] - cum_hist[25])   #per pixel error value range in 26-255
var = np.var(img, ddof=1)


print("FN: \n preciseRateStrict:", preciseRateStrict,  "\n mean error for strict errors ", strictErrorMean, "\npreciserateRough : ",preciseRateRough,
          "\n mean error for rough errors ", roughErrorMean, "\n global mean error:", meanError,'\n varience:', var)