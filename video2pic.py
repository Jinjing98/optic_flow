import cv2


def getFrame(videoPath, svPath,expected_size):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0 #此值是图片名称的计数起始。
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:

                frame = cv2.resize(frame, expected_size, interpolation=cv2.INTER_LINEAR)


                # cv2.imshow('video', frame)
                numFrame += 1

                newPath = svPath + str(numFrame) + ".png" #此处的图片后缀可根据需要更改
                cv2.imencode('.png', frame)[1].tofile(newPath)
        if cv2.waitKey(10) == 27:
            break

if __name__ == '__main__':
    # videoPath='D:\Study\Datasets\\video64.mp4' #要处理的视频文件名
    # savePicturePath='D:\Study\Datasets\\video64\\' #处理之后图片的保存路径
    # crop_size = (512, 384)

    videoPath = "D:\Study\Datasets\\moreCam.mp4"
    savePicturePath = "D:\Study\Datasets\\moreCam\\ori\\"
    crop_size = (720,288)


    getFrame(videoPath,savePicturePath,crop_size)
