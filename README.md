This repo is about my research project "Optical flow estimation in surgery videos".



In this repo, there are four different implementation of optical flow estimation. 

Three of them belongs to traditional differential calculation methods(Lucas Kanade, HornSchunck, Gunnar Farnback), the specific implementation are located in the three files : hornschunck_real.py, gunnarfarnback_real.py , and sparse_lk_real.py.   



The forth method is a pretrained CNN  model "flownet2".

(source: https://github.com/NVIDIA/flownet2-pytorch) I tested this our surgery video on the model.



Besides, I wrote some python scripts to assist the video processing to achieve purposes such as video cropping, video clipping, add noise,  the transformations between videos and frames etc.



What is more, I wrote some python scripts in order to quantitatively evaluate the results under different methods.   Which include warp.py to warp the one frame to the its next frame based on the optic flow estimation. Also, in signalProcess.py, I implement FFT(fast Fourier transformation) to get the frequency of heart pulse.