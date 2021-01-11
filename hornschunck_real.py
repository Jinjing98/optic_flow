# Importing libraries
import cv2
import math
import time
import  numpy as np
from pyoptflow import HornSchunck, getimgfiles
from frames2video import pic2video


size = (854,480)
# path = "D:\Study\Datasets\HS_test\HS_test_result\\"
flow_path = "D:\Study\Datasets\\test3L\HSflow\\"
# Capturing the video file 0 for videocam    else you can provide the url

# cap =  cv2.VideoCapture("D:\Study\Datasets\\video64.mp4")  # video64 68
# cap = cv2.VideoCapture("D:\Study\Datasets\\noisy64.mp4")

cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("D:\Study\Datasets\\test3L.mp4")
# cap = cv2.VideoCapture("D:\Study\Datasets\\CamPlusOrgan.mp4")
# cap = cv2.VideoCapture("D:\Study\Datasets\\Cam.mp4")

# def sigmoid(x):
#      return 1 / (1 + math.e ** -x)


# Reading the first frame
_, frame1 = cap.read()
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
rows, cols = prvs.shape
#prvs = cv2.medianBlur(prvs, 5)
# for i in range(100000):
#     x = np.random.randint(0, rows)
#     y = np.random.randint(0, cols)
#     prvs[x, y] = 255
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value   HSV 8UC则取值范围是h为0-180、s取值为0-255、v取值是0-255.

hsv_mask[..., 1] = 255

SIGS_ang = []
SIGS_mag = []
# SIGS_gray = []
# SIGS_blue = []
# SIGS_U = []
# SIGS_V = []


i = 1
# Till you scan the video
while(i):


	# Capture another frame and convert to gray scale
	ret, frame2 = cap.read()
	# if i < 4000:  #   直接从好帧开始运行
	#     i += 1
	#     continue


	if ret != True:
		break
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	#next = cv2.medianBlur(next, 5)
	# for i in range(0):
	# 	x = np.random.randint(0, rows)
	# 	y = np.random.randint(0, cols)
	# 	prvs[x, y] = 255
	# next = cv2.medianBlur(next, 5)
	##next = cv2.medianBlur(next, 5)
	# Computes a dense optical flow using the Gunnar Farneback's algorithm.
	start = time.time()
	#flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=5, iterations=5, poly_n=5,
	#									   poly_sigma=1.1, flags=0)

	U, V = HornSchunck(prvs, next, alpha=10, Niter=10)
	end = time.time()
	print("flow computing time: " + str(end-start))
	# Compute magnite and angle of 2D vector
	# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees= False)   #in radians not degrees
	start = time.time()

	mag, ang = cv2.cartToPolar(U,V,angleInDegrees= False)   #in radians not degrees

	# Set image hue according to the angle of optical flow   HUE  1st d of hsv
	hsv_mask[..., 0] = ang * 180 / np.pi / 2
	# Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
	hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)   #nolinear transformation of value   sigmoid?
	# var = hsv_mask[..., 2] - mag
	end = time.time()
	print('visualization '+ str(end-start))

	# another way  of normolize maginitude   use mag**2 is better than mag since this way can enlarge large movement and filter out minor movement

	SIG_ang = np.mean(hsv_mask[..., 0])
	SIG_mag = np.mean(hsv_mask[..., 2])
	# SIGS_B = np.mean()
	SIGS_ang.append(SIG_ang)
	SIGS_mag.append(SIG_mag)

	# SIG_gray = np.mean(next)
	# SIG_blue = np.mean(frame2[..., 0])
	# SIGS_gray.append(SIG_gray)
	# SIGS_blue.append(SIG_blue)

	# Convert to rgb


	rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
	double_representation = cv2.addWeighted(rgb_representation,3,frame2, 0.5, 0)




	# cv2.imshow('result_window', double_representation)
	cv2.imshow('result_window', rgb_representation)
	# cv2.imwrite(path + str(i) + '.png', double_representation)
	# cv2.imwrite(flow_path + str(i) + '.png', rgb_representation)

	kk = cv2.waitKey(20) & 0xff
	# Press 'e' to exit the video
	if kk == ord('e'):
		break
	# Press 's' to save the video
	elif kk == ord('s'):
		cv2.imwrite('Optical_image.png', frame2)
		cv2.imwrite('HSV_converted_image.png', rgb_representation)
	prvs = next
	i += 1
# pic2video(path,size)
cap.release()
cv2.destroyAllWindows()
print(np.array(SIGS_ang).size)
print(np.array(SIGS_mag).size)
# np.save(flow_path+"SIGS_ang.txt",SIGS_ang)
# np.save(flow_path+"SIGS_mag.txt",SIGS_mag)

