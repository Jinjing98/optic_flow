from HS_GF_FN_func import HS,GF
from signalProcess import fft


videopath = 'D:\Study\Datasets\signalVideos\staticCam\\resp1.avi'
sigpath = 'D:\Study\Datasets\signalVideos\staticCam\\resp1\\'
flow_path = "D:\Study\Datasets\signalVideos\staticCam\\resp1\\GF\\"

# videopath = 'D:\Study\Datasets\signalVideos\staticCam\\card1.avi'
# sigpath = 'D:\Study\Datasets\signalVideos\staticCam\\card1\\'
# flow_path = "D:\Study\Datasets\signalVideos\staticCam\\card1\\GF\\"

videopath = 'D:\Study\Datasets\signalVideos\staticCam\\resp5_turbB1.avi'
sigpath = 'D:\Study\Datasets\signalVideos\staticCam\\resp5_turbB1\\'
flow_path = "D:\Study\Datasets\signalVideos\staticCam\\resp5_turbB1\\GF\\"
#
videopath = 'D:\Study\Datasets\signalVideos\staticCam\\resp5_turbG1.avi'
sigpath = 'D:\Study\Datasets\signalVideos\staticCam\\resp5_turbG1\\'
flow_path = "D:\Study\Datasets\signalVideos\staticCam\\resp5_turbG1\\GF\\"
# videopath = 'D:\Study\Datasets\signalVideos\staticCam\\cardNresp_turbB1.avi'
# sigpath = 'D:\Study\Datasets\signalVideos\staticCam\\cardNresp_turbB1\\'
# flow_path = "D:\Study\Datasets\signalVideos\staticCam\\cardNresp_turbB1\\GF\\"
videopath = 'D:\Study\Datasets\moreCamBestNontrembling.mp4'
sigpath = 'D:\Study\Datasets\moreCamBestNontrembling\\'
flow_path = "D:\Study\Datasets\moreCamBestNontrembling\HS\\"

sigpath = 'D:\Study\Datasets\moreCamBest\\'



# HS(videopath,flow_path,sigpath,[0,28])  #  调用这种函数必须从0开始  否则fft 必须从0开始（持续Δt）
# GF(videopath,flow_path,sigpath,[0,28])
# fft(25,[0,90],sigpath+'GFSIGS_mag.txt.npy',5, "detail",True,[1.4,1.6])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
# fft(25,[15,30],sigpath+'SIGS_gray.txt.npy',5, "detail",True,[0.1,0.3])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
# fft(25,[0,25],sigpath+'GFSIGS_mag.txt.npy',5, "detail",True,[0.2,0.4])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
fft(25,[0,20],sigpath+'SIGS_gray.txt.npy',5, "detail",True,[0.2,0.4])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
# fft(25,[1,16],sigpath+'SIGS_graybetter.txt.npy',5, "detail",True,[0.2,0.4])   #max_freq 5 必须是 25 的因数  25/2.5 must be int
