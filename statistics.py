import matplotlib.pyplot as plt
import numpy as np
import cv2


path =  "D:\Study\Datasets\pairs\\video64\\diff154NN.png"
pathHS =  "D:\Study\Datasets\pairs\\video64\\diff154HS.png"
pathGF =  "D:\Study\Datasets\pairs\\video64\\diff154GF.png"
img = cv2.imread(path)
imgHS = cv2.imread(pathHS)
imgGF = cv2.imread(pathGF)

plt.hist(img.ravel(), 256, [0, 256])
plt.show()
x = np.arange(256)
totalNum = 512*384

hist = cv2.calcHist([img], [0], None, [256], [0, 256])    #   每16个像素当作一个单元  共256个条条  选公因数
hist = hist.flatten()
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
plt.subplot(2,1,1)
plt.plot(x[0:30],(cum_hist/totalNum)[0:30],alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
plt.plot(x[0:30],(cum_histHS/totalNum)[0:30],alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
plt.plot(x[0:30],(cum_histGF/totalNum)[0:30],alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
plt.legend(loc='upper left', bbox_to_anchor=(0.85, 0.55))
plt.ylabel('probability')
plt.xlabel('error')
plt.title('\n'+"CDF of error range in [0,30]")

plt.subplot(2,1,2)
plt.plot(x[-30:],(cum_hist/totalNum)[-30:],alpha=0.5,color="r", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'NN')
plt.plot(x[-30:],(cum_histHS/totalNum)[-30:],alpha=0.5,color="g", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'HS')
plt.plot(x[-30:],(cum_histGF/totalNum)[-30:],alpha=0.5,color="b", linestyle="--",marker = "x", markersize=4,linewidth = 1,label= 'GF')
# plt.scatter(x,hist/totalNum,s = 4, c = 'r',marker = "o",linewidths = 0.1,label= 'NN')
# plt.scatter(x,histHS/totalNum,s = 4, c = 'g',marker = "x",linewidths = 0.1,label= 'HS')
# plt.plot(x, y1, color="r", linestyle="-", marker="^", linewidth=1)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.55))
plt.ylabel('probability')
plt.xlabel('error')
plt.title( '\n'+"CDF of error range in [225,255]")
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

# plt.step(x,cum_hist/totalNum,color="b",lw=1)
# plt.title("cdf of error")
# plt.grid()
# plt.show()












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


meanError = np.mean(imgGF)
# 这两个指标定义error出现的数量
preciseRateStrict = cum_histGF[0]/totalNum     #define a error when it is not zero   strict error
preciseRateRough = cum_histGF[25]/totalNum     #define error when the error is bigger than 16 pixels

# 这几个指标定义error 偏离的程度  （与error值有关）
meanError = np.mean(imgGF)
strictErrorMean = np.sum(histGF[1:-1] * x[1:-1])/(cum_histGF[-1] - cum_histGF[0])   #per pixel error value range in 1-255
roughErrorMean = np.sum(histGF[26:-1] * x[26:-1])/(cum_histGF[-1] - cum_histGF[25])   #per pixel error value range in 26-255


print("preciseRateStrict:", preciseRateStrict,  "\n mean error for strict errors ", strictErrorMean, "\npreciserateRough : ",preciseRateRough,
          "\n mean error for rough errors ", roughErrorMean, "\n global mean error:", meanError,)


