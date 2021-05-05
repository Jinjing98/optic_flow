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
		cv2.imshow('result_window', rgb_representation)
		#cv2.imwrite(flow_path + str(i) + '.png', rgb_representation)

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


def GF_window(videopath,sigpath,ranges,gridnumX,gridnumY):
	cap = cv2.VideoCapture(videopath)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	_, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

	lower =  fps*ranges[0]
	upper =  fps*ranges[1]
	framenum = upper - lower
	hsv_mask = np.zeros_like(frame1)
	hsv_mask[..., 1] = 255

	gridWidth = int(width / gridnumX)  # int 0.6 = 0; 320  required to be zhengchu, but if not, then directly negelact the boundary
	gridHeight = int(height / gridnumY)  # 240
	SIGS_gray = np.zeros((framenum,gridnumY,gridnumX))#4,3
	SIGS_ang = np.zeros((framenum,gridnumY,gridnumX))
	SIGS_mag = np.zeros((framenum,gridnumY,gridnumX))


	i = 1
	timepoint = 0
	# Till you scan the video
	while (i):
		start1 = time.time()
		# Capture another frame and convert to gray scale
		ret, frame2 = cap.read()
		if i < lower:  #   直接从好帧开始运行
		    i += 1
		    continue
		if i >= upper:
			break
		if ret != True:
			break
		next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



		start = time.time()
		flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=5, iterations=5, poly_n=5,
											   poly_sigma=1.1, flags=0)
		end = time.time()
		print("flow computing time: " + str(end - start))

		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees= False)   #in radians not degrees
		hsv_mask[..., 0] = ang * 180 / np.pi / 2
		# Set value as per the normalized magnitude of optical flow      Value 3rd d of hsv
		hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255,
										 cv2.NORM_MINMAX)  # nolinear transformation of value   sigmoid?
		# var = hsv_mask[..., 2] - mag

		start = time.time()
		# for idx_x in range(gridnumX):#how to speed up when it is getting finner
		# 	for idx_y in range(gridnumY):
		# 		#print(idx_y,idx_x)
		# 		#print(idx_y*gridHeight,(idx_y+1)*gridHeight, idx_x*gridWidth,(idx_x+1)*gridWidth)
		#
		# 		win_img = next[idx_y*gridHeight:(idx_y+1)*gridHeight, idx_x*gridWidth:(idx_x+1)*gridWidth]
		#
		# 		#print(np.mean(win_img))
		# 		#print(idx_y,idx_x)
		# 		SIGS_gray[timepoint,idx_y,idx_x] = np.mean(win_img)
		# 		SIGS_ang[timepoint,idx_y,idx_x] = np.mean(hsv_mask[..., 0])
		# 		SIGS_mag[timepoint,idx_y,idx_x] = np.mean(hsv_mask[..., 2])
		#before it took around 3 mins, now it is way faster based on numpy vectorization
		#https://codingdict.com/questions/179693
		#print(next)
		next = next.reshape(gridnumY,gridHeight,gridnumX,gridWidth)
		next_ang = hsv_mask[..., 0].reshape(gridnumY,gridHeight,gridnumX,gridWidth)
		next_mag = hsv_mask[..., 2].reshape(gridnumY,gridHeight,gridnumX,gridWidth)
		#print(next[0,:,0,:])

		SIGS_gray[timepoint] = next.mean(axis=(1, 3))  # y.mean(axis=(1,3))
		# print(meanvalue.shape)#48 72
		SIGS_ang[timepoint] = next_ang.mean(axis=(1, 3))
		SIGS_mag[timepoint] = next_mag.mean(axis=(1, 3))




		end = time.time()
		print('mean signal collections for windows' + str(end - start))



				#cv2.imshow("cropped", win_img)
				#cv2.waitKey(0)
		timepoint += 1
		#print(SIGS_gray[timepoint-1])
		i += 1
		end1 = time.time()
		print("total time: " + str(end1 - start1))

	#print(SIGS_gray[:, 1, 1])  # the list for block 1,1
	print(np.array(SIGS_ang).size)
	print(np.array(SIGS_mag).size)
	#np.save(sigpath+"GFSIGS_ang.txt",SIGS_ang)
	#np.save(sigpath+"GFSIGS_mag.txt",SIGS_mag)
	np.save(sigpath,SIGS_gray)



#GF_window("D:\Study\Datasets\signalVideos\staticCam\card1.avi","","",[0,1],8,8)  # WE DO NOT NEED TO COMPUTE FOR EACH FRAME WHEN COLLECT ARRAY DATA

