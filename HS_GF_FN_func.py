import cv2
import math
import time
import  numpy as np
from pyoptflow import HornSchunck, getimgfiles
from frames2video import pic2video


def HS(videopath,flow_path,sigpath,range):
	size = (854, 480)

	cap = cv2.VideoCapture(videopath)
	_, frame1 = cap.read()
	# Convert to gray scale
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	lower =  fps*range[0]
	upper =  fps*range[1]
	hsv_mask = np.zeros_like(frame1)

	hsv_mask[..., 1] = 255

	SIGS_ang = []
	SIGS_mag = []
	SIGS_gray = []
	SIGS_graybetter = []
	# SIGS_U = []
	# SIGS_V = []

	i = 1
	# Till you scan the video
	while (i):

		# Capture another frame and convert to gray scale
		ret, frame2 = cap.read()

		if i < lower:  #   直接从好帧开始运行
		    i += 1
		    continue
		if i > upper:
			break

		if ret != True:
			break
		next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		start = time.time()
		# flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=5, iterations=5, poly_n=5,
		#									   poly_sigma=1.1, flags=0)

		U, V = HornSchunck(prvs, next, alpha=10, Niter=10)
		end = time.time()
		print("flow computing time: " + str(end - start))
		# Compute magnite and angle of 2D vector
		# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees= False)   #in radians not degrees
		start = time.time()
		mag, ang = cv2.cartToPolar(U, V, angleInDegrees=False)  # in radians not degrees

		# Set image hue according to the angle of optical flow   HUE  1st d of hsv
		hsv_mask[..., 0] = ang * 180 / np.pi / 2
		# Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
		hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255,
										 cv2.NORM_MINMAX)  # nolinear transformation of value   sigmoid?
		# var = hsv_mask[..., 2] - mag
		end = time.time()
		print('visualization ' + str(end - start))


		SIG_ang = np.mean(hsv_mask[..., 0])
		SIG_mag = np.mean(hsv_mask[..., 2])
		SIG_gray = np.mean(next)# SIGS_B = np.mean()
		SIGS_ang.append(SIG_ang)
		SIGS_mag.append(SIG_mag)
		SIGS_gray.append(SIG_gray)
		SIG_graybetter = np.true_divide(next.sum(), (next != 0).sum())  # 对二维度灰度图中所有非零元素取均值
		SIGS_graybetter.append(SIG_graybetter)

		if i == 300:
			print("jinjing")

		rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
		double_representation = cv2.addWeighted(rgb_representation, 3, frame2, 0.5, 0)

		cv2.imshow('result_window', double_representation)
		# cv2.imshow('result_window', rgb_representation)
		cv2.imwrite(flow_path + str(i) + '.png', rgb_representation)

		kk = cv2.waitKey(20) & 0xff
		# Press 'e' to exit the video
		if kk == ord('e'):
			break
		prvs = next
		i += 1
	# pic2video(path,size)
	cap.release()
	cv2.destroyAllWindows()
	print(np.array(SIGS_ang).size)
	print(np.array(SIGS_mag).size)
	np.save(sigpath+"HSSIGS_ang.txt",SIGS_ang)
	np.save(sigpath+"HSSIGS_mag.txt",SIGS_mag)
	np.save(sigpath+"SIGS_gray.txt",SIGS_gray)
	np.save(sigpath+"SIGS_graybetter.txt",SIGS_graybetter)



def GF(videopath,flow_path,sigpath,range):
	size = (854, 480)

	cap = cv2.VideoCapture(videopath)
	_, frame1 = cap.read()
	# Convert to gray scale
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	lower =  fps*range[0]
	upper =  fps*range[1]
	hsv_mask = np.zeros_like(frame1)

	hsv_mask[..., 1] = 255

	SIGS_ang = []
	SIGS_mag = []
	SIGS_gray = []
	SIGS_graybetter = []
	# SIGS_U = []
	# SIGS_V = []

	i = 1
	# Till you scan the video
	while (i):

		# Capture another frame and convert to gray scale
		ret, frame2 = cap.read()

		if i < lower:  #   直接从好帧开始运行
		    i += 1
		    continue
		if i > upper:
			break

		if ret != True:
			break
		next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		start = time.time()
		flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=5, iterations=5, poly_n=5,
											   poly_sigma=1.1, flags=0)
		end = time.time()
		print("flow computing time: " + str(end - start))
		# U, V = HornSchunck(prvs, next, alpha=10, Niter=10)

		# Compute magnite and angle of 2D vector
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees= False)   #in radians not degrees

		start = time.time()
		# mag, ang = cv2.cartToPolar(U, V, angleInDegrees=False)  # in radians not degrees

		# Set image hue according to the angle of optical flow   HUE  1st d of hsv
		hsv_mask[..., 0] = ang * 180 / np.pi / 2
		# Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
		hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255,
										 cv2.NORM_MINMAX)  # nolinear transformation of value   sigmoid?
		# var = hsv_mask[..., 2] - mag
		end = time.time()
		print('visualization ' + str(end - start))


		SIG_ang = np.mean(hsv_mask[..., 0])
		SIG_mag = np.mean(hsv_mask[..., 2])
		SIG_gray = np.mean(next)# SIGS_B = np.mean()
		SIGS_ang.append(SIG_ang)
		SIGS_mag.append(SIG_mag)
		SIGS_gray.append(SIG_gray)
		SIG_graybetter = np.true_divide(next.sum(), (next != 0).sum())  # 对二维度灰度图中所有非零元素取均值
		SIGS_graybetter.append(SIG_graybetter)

		if i == 300:
			print("jinjing")

		rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
		double_representation = cv2.addWeighted(rgb_representation, 3, frame2, 0.5, 0)

		cv2.imshow('result_window', double_representation)
		# cv2.imshow('result_window', rgb_representation)
		cv2.imwrite(flow_path + str(i) + '.png', rgb_representation)

		kk = cv2.waitKey(20) & 0xff
		# Press 'e' to exit the video
		if kk == ord('e'):
			break
		prvs = next
		i += 1
	# pic2video(path,size)
	cap.release()
	cv2.destroyAllWindows()
	print(np.array(SIGS_ang).size)
	print(np.array(SIGS_mag).size)
	np.save(sigpath+"GFSIGS_ang.txt",SIGS_ang)
	np.save(sigpath+"GFSIGS_mag.txt",SIGS_mag)
	np.save(sigpath+"SIGS_gray.txt",SIGS_gray)
	np.save(sigpath+"SIGS_graybetter.txt",SIGS_graybetter)
