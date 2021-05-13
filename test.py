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

cnums = 3 + 1j * np.arange(6,11)
X = [x.real for x in cnums]
Y = [x.imag for x in cnums]
plt.scatter(Y,X, color='red')
plt.show()



#x_axix，train_pn_dis这些都是长度相同的list()
x_axix = [i for i in range(90)]
array3d = np.load("D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\\arti_videos\BinW127\gray\size854_480\SIGS_gray854_480.npy")
train1 = array3d[:,240,425][:90]
train2 = array3d[:,50,425][:90]
train3 = array3d[:,45,21][:90]
train4 = array3d[:,30,77][:90]

# train5 = array3d[:,42,21][:150]
# train6 = array3d[:,43,21][:150]
train7 = array3d[:,48,21][:90]
train8 = array3d[:,275,356][:90]
print(2)

#开始画图
sub_axix = filter(lambda x:x%200 == 0, x_axix)
plt.title('Result Analysis')
plt.plot(x_axix, train1, color='green', label='21,44 FN1Bound')
plt.plot(x_axix, train2, color='red', label='77,29 FN(PEPPER)')
# plt.plot(x_axix, train4, color='black', label='77,30 TP(CORRECT)')
# # plt.plot(x_axix, train3,  color='green', label='21,45 FN2Bound')
# # plt.plot(x_axix, train4, color='blue', label='4')
# #
# # plt.plot(x_axix, train5, color='black', label='5')
# # plt.plot(x_axix, train6, color='black', label='6')
# plt.plot(x_axix, train7,  color='black', label='21,48 TP(CORRECT)')
# plt.plot(x_axix, train8, color='blue', label='356,275 FP')
plt.legend() # 显示图例

plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()