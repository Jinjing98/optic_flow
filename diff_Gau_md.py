# -*- coding: utf-8 -*-
#https://github.com/methylDragon/opencv-motion-detector/blob/master/Motion%20Detector.py
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

FRAMES_TO_PERSIST = 10000000






videoname = 2
prefix = "D:\Study\Datasets\AAAtest\\"
prefix4mask = prefix+str(2)+"\\"
videopath =  prefix+str(videoname)+".avi"





cap = cv2.VideoCapture(videopath)
# Init frame variables
first_frame = None
next_frame = None

# Init display font and timeout counters
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0
i = 0

# LOOP!
while True:

    # Set transient motion detected as false
    transient_movement_flag = False

    # Read frame
    ret, frame = cap.read()
    text = "Unoccupied"

    if not ret:
        break
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # If the first frame is nothing, initialise it
    if first_frame is None: first_frame = gray

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appriopriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

    # Set the next frame to compare (the current frame)
    next_frame = gray

    # Compare the two frames, find the difference
    frame_delta = cv2.absdiff(first_frame, next_frame)
    _,thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)


    # Fill in holes via dilate(), and find contours of the thesholds
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imshow("",thresh)
    cv2.waitKey(0)

    kk = cv2.waitKey(20) & 0xff
    if kk == ord('q'):
        break
    # Press 's' to save the video
    elif kk == ord('s'):
        pass

        path_map = prefix4mask+"vibe_"+str(i)+".png"
        cv2.imwrite(path_map,thresh)
    i += 1

