import cv2

INPUT_FILE = 't1.avi'
INPUT_FILE = "D:\Study\Datasets\\moreCamStable.mp4"

OUTPUT_FILE = 'D:\Study\Datasets\\moreCamBest.avi'
start_frame = 0*25
end_frame = 29*25  # 1:15 - 1:40
# start_frame = 75*25
# end_frame = 100*25  # 1:15 - 1:40
reader = cv2.VideoCapture(INPUT_FILE)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = reader.get(cv2.CAP_PROP_FPS)
fps = 25







writer = cv2.VideoWriter(OUTPUT_FILE,
                         cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                         fps,  # fps
                         (125,95))  # resolution   (原本为 1440，288)

print(reader.isOpened())
have_more_frame = True
c = 0
while have_more_frame:
    have_more_frame, frame = reader.read()
    c += 1
    if c >= start_frame and c <= end_frame:
        cv2.waitKey(1)
        frame = frame[81:175, 291:415]

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