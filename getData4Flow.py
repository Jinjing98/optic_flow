import cv2
import os


path4videos = 'D:\Study\Datasets\cholec80\\videos\\'
path4current = 'D:\Study\Datasets\cholec80\current\\'
path4next = 'D:\Study\Datasets\cholec80\\next\\'
# path4videos = 'D:\Study\Datasets\hamlyn\\videos\\'
# path4current = 'D:\Study\Datasets\hamlyn\current\\'
# path4next = 'D:\Study\Datasets\hamlyn\\next\\'
videosPathList = []

def getdata(path,name):
    reader = cv2.VideoCapture(path)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameIdx = 0
    have_more_frame = True
    domain = [250, 2500, 5000, 10000, 20000, 30000, 40000]  #for cholec80
    # domain = [50, 100, 150, 200, 250]
    k = 0
    while have_more_frame:
        have_more_frame, frame = reader.read()
        frameIdx += 1
        if frameIdx in domain:
            k += 1
            # if name[0] == 'c':
            #     frame = frame[:height, 10:int(width / 2 - 10)]
            #     frame = frame[:256, :512]
            # elif name[:7] == 'stereo1':
            #     frame = frame[:192,:256]
            # else:
            #     frame = frame[:256, 30:286]

            frame = frame[:384,:768]
            cv2.imshow('', frame)
            cv2.waitKey(1)
            cv2.imwrite(path4current + name[:-4] +'_'+str(k)+ '.png', frame)
            for i in range(3):
                have_more_frame, frame = reader.read()
                frameIdx += 1
                # if name[0] == 'c':
                #     frame = frame[:height, 10:int(width / 2 - 10)]
                #     frame = frame[:256, :512]
                # elif name[:7] == 'stereo1':
                #     frame = frame[:192, :256]
                # else:
                #     frame = frame[:256, 30:286]
                frame = frame[:384, :768]

            cv2.imwrite(path4next + name[:-4] +'_'+str(k)+ '.png', frame)

            if k == 7:
                break

    reader.release()
    cv2.destroyAllWindows()


for f_name in os.listdir(path4videos):
    if  f_name.endswith('.mp4'):
        videosPathList.append(f_name)

for name in videosPathList:
    path = path4videos + name
    getdata(path,name)








