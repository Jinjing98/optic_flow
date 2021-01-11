import cv2

INPUT_FILE = 't1.avi'
INPUT_FILE = "D:\Study\Datasets\\CamPlusOrgan.mp4"

OUTPUT_FILE = 'D:\Study\Datasets\\moreCam.avi'
start_frame = 164*25
end_frame = 194*25  # 1:15 - 1:40
# start_frame = 75*25
# end_frame = 100*25  # 1:15 - 1:40
reader = cv2.VideoCapture(INPUT_FILE)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = reader.get(cv2.CAP_PROP_FPS)







writer = cv2.VideoWriter(OUTPUT_FILE,
                         cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                         fps,  # fps
                         (720,288))  # resolution

print(reader.isOpened())
have_more_frame = True
c = 0
while have_more_frame:
    have_more_frame, frame = reader.read()
    c += 1
    if c >= start_frame and c <= end_frame:
        cv2.waitKey(1)
        frame = frame[0:288, 0:720]

        writer.write(frame)
        print(str(c) + ' is ok')
    if c > end_frame:
        print('completely!')
        break

    kk = cv2.waitKey(20) & 0xff
    if kk == ord('e'):
        break



writer.release()
reader.release()
cv2.destroyAllWindows()