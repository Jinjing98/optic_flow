import cv2
import numpy as np
import frames2video

j = 1
while 1:
      string = "D:\Study\Datasets\\video64\\"+str(j)+".png"
      string2 = "D:\Study\Datasets\\video64_noise\\" + str(j) + ".png"
      print(string)
      img = cv2.imread(string)
      j += 1
      rows,cols,dims=img.shape

      for i in range(5000):

        x=np.random.randint(0,rows)

        y=np.random.randint(0,cols)

        img[x,y,:]=255
      cv2.imshow("",img)
      cv2.imwrite(string2,img)

pic2video()