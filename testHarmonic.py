import numpy as np
import pylab as pl
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import time
import pylab as plt
from scipy import stats
from HS_GF_FN_GRAY_func import HS,GF,GRAY,FN
from test import draw2pointsRAW_FFT,on_EVENT_LBUTTONDOWN


sampleN = 500

t = np.array([0.001*i for i in range(sampleN)])

X = 50*np.sin(2 * np.pi * 50 * t)

x1 = 30*np.sin(2*np.pi*100*t + 30*(np.pi/180))
x2 = 10*np.sin(2*np.pi*200*t + 50*(np.pi/180))
x3 = 5*np.sin(2*np.pi*300*t + 90*(np.pi/180))

Xt = X + x1 + x2 + x3+20
fft = np.abs(np.absolute(np.fft.fft((Xt)) )* 2 /sampleN)
f = [1*i/0.05 for i in range(sampleN)]
plt.plot(t,Xt )
# plt.plot(f,fft)

plt.show()