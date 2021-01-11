import flowiz as fz
import glob
import matplotlib. pyplot as plt
import cv2

files = glob.glob('D:/Study/flownet2-pytorch/out2/inference/run.epoch-0-flow-field/000001.flo')
img = fz.convert_from_file(files[0])
# cv2.imshow("",img)
# cv2.waitKey(0)
plt.imshow(img)
plt.show()
