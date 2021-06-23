import numpy as np
import cv2
import os
from pandas import DataFrame
import csv

from pathlib import Path
import os,re
from glob import glob
#
# dir4masksets = "D:\Study\Datasets\AATEST\\new_short\\15"
# masksetpaths = [y for x in os.walk(dir4masksets) for y in glob(os.path.join(x[0], '*.npy'))]
#
# # list(Path(path4video).rglob("*maskSet.npy"))

# for masksetpath in masksetpaths:
#     head = os.path.split(masksetpath)[0]#[-21:]  # videoname+"\\"+mode+"\\"+note+"\size"+String0
#     tail = os.path.split(masksetpath)[1][:-12]
# # print(re.findall(r"a(.*)b", str))
#     print("head",head)
#     print("tail",tail)
#     print("ONE: ",re.findall(r"short\\(.+?)\\size", head))
#     print("TWO: ",re.findall(r"size(.*)", head))
    # ROW = [masksetpath].append(generate_stats(imgx, num4indexX, imgy, num4indexY, masksetpath, truthimg_path))
#
# a = ["jinjing","zl","kk","m"]
# b = [1,2,3,4]
# c = [11,2,22,33]
# df = DataFrame({"name":a,"age":b,"size":c})
#
# print(df)
# df.to_excel("test.xlsx",index=False,sheet_name='FN ang 1*1')

def generate_stats(imgx,num4indexX,imgy,num4indexY,masksetpath,stage,truthimg_path):# stage = 0/1/2/3
    mask0 = np.load(masksetpath)[stage].astype(bool)

    gridwidth = int(imgx / num4indexX)
    gridheight = int(imgy / num4indexY)
    # mask0 = maskNfreqID_infoMat[i]
    y_idx = np.nonzero(mask0)[0]
    x_idx = np.nonzero(mask0)[1]
    mask0 = np.zeros((imgy, imgx))
    for y, x in zip(y_idx, x_idx):
        mask0[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask0 = mask0.astype(bool)


    truth_mask = cv2.imread(truthimg_path)[:, :, 0].astype(bool)
    TP_mask = np.logical_and(truth_mask, mask0)
    TN_mask = np.logical_and(1 - truth_mask, 1 - mask0)
    FP_mask = np.logical_and(1 - truth_mask, mask0)
    FN_mask = np.logical_and(truth_mask, 1 - mask0)

    TP = np.count_nonzero(TP_mask)
    FP = np.count_nonzero(FP_mask)
    TN = np.count_nonzero(TN_mask)
    FN = np.count_nonzero(FN_mask)
    Num_pridic_P = TP+FP
    Num_pridic_N = TN+FN
    Num_p = TP+FN#np.count_nonzero(truth_mask)
    Num_n = TN+FP#np.count_nonzero(1 - truth_mask)
    Total = Num_p+Num_n

    precision =round( TP/Num_pridic_P,3 )if Num_pridic_P != 0 else None#np.count_nonzero(TP_mask)/(np.count_nonzero(TP_mask)+np.count_nonzero(FP_mask))
    recall = round(TP/Num_p,3 )if Num_p != 0 else None#np.count_nonzero(TP_mask)/(np.count_nonzero(TP_mask)+np.count_nonzero(FN_mask))
    F1 = round(2*precision*recall/(precision+recall),3 )if precision!= None and recall != None and (precision+recall !=0) else None
# specificity, false positive rate (FPR), false negative rate (FNR), percentage of wrong classification (PWC)
    FPR = round(FP/Num_n,3) if Num_n != 0 else None
    FNR = round(FN/Num_p,3) if Num_p != 0 else None
    PWC = round((FP+FN)/Total,3)
    ID = masksetpath[:100]
    # specificity = TN/Num_n
    return [Num_p,Num_n,TP,TN,FP,FN,precision,recall,F1,FPR,FNR,PWC]




def generateALLstats4video(imgx,imgy,truthimg_path,path4csv,dir4masksets):
    # field names
    fields = ["methodNmode","numX","numY","mask_stage","P","N","TP","TN","FP","FN","params","precision", "recall", "F1", "FPR", "FNR", "PWC"]

    with open(path4csv, 'a') as csvfile:  #  "w" will remove history
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        # csvwriter.writerow(fields)

        #recursive find
        masksetpaths =[y for x in os.walk(dir4masksets) for y in glob(os.path.join(x[0], '*maskSet.npy'))]

        #list(Path(path4video).rglob("*maskSet.npy"))

        for masksetpath in masksetpaths:

            head = os.path.split(masksetpath)[0]  #videoname+"\\"+mode+"\\"+note+"\size"+String0
            modeNnote = re.findall(r"Datasets\\(.+?)\\size", head)
            size = re.findall(r"size(.*)", head)
            params_tail = os.path.split(masksetpath)[1][:-12]
            # print(re.findall(r"size(.*)_", head))
            # print(re.findall(r"_(.*)", head)[0])
            num4indexX = int(re.findall(r"size(.*)_", head)[0])
            num4indexY = int(re.findall(r"_(.*)", head)[0])
            #  stage can be 0 1 2 3, stage 3 is the final mask
            for stage in [0,1,2,3]:
                LIST = generate_stats(imgx, num4indexX, imgy, num4indexY, masksetpath, stage, truthimg_path)
                ROW = [modeNnote[0], num4indexX,num4indexY, params_tail, stage] + LIST
                print(ROW)
                csvwriter.writerow(ROW)













































































































































































# truth_path = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6\GF\mag\size512_384\\truth.png"
# path ="D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6\GF\mag\size512_384\\512_384_5_15_1_0.11_0.85_infoMat_mask_ID.npy"
# maskNfreqID_infoMat = np.load(path)
# mask0 = maskNfreqID_infoMat[0].astype(bool)
# mask25 = maskNfreqID_infoMat[2].astype(bool)
# mask50 = maskNfreqID_infoMat[3].astype(bool)
# mask75 = maskNfreqID_infoMat[4].astype(bool)
# FN = 0
# TN = 0
# FP = 0
# TP = 0

#
# truth_mask = cv2.imread(truth_path)[:,:,0].astype(bool)
# Num_p = np.count_nonzero(truth_mask)
# Num_n = np.count_nonzero(1-truth_mask)
#
#
# # cv2.imshow("truth_mask",truth_mask.astype(np.uint8)*255)
# # cv2.waitKey(0)
#
#
# TP_mask = np.logical_and(truth_mask,mask0)
# TN_mask = np.logical_and(1-truth_mask,1-mask0)
# FP_mask = np.logical_and(1-truth_mask,mask0)
# FN_mask = np.logical_and(truth_mask,1-mask0)
#
# cv2.imshow("TP_mask",TP_mask.astype(np.uint8)*255)
# cv2.waitKey(0)
# cv2.imshow("TN_mask",TN_mask.astype(np.uint8)*255)
# cv2.waitKey(0)
# cv2.imshow("FP_mask",FP_mask.astype(np.uint8)*255)
# cv2.waitKey(0)
# cv2.imshow("FN_mask",FN_mask.astype(np.uint8)*255)
# cv2.waitKey(0)
#
# gt_img_path= "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6\GF\mag\size512_384\\truth.png"
# dir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6\GF\mag\size512_384\\"
# items = os.listdir(dir)
# infoMATlist = []
# for names in items:
#     if names.endswith("ID.npy"):
#         infoMATpath = dir+names
#         TP,FP,TN,FN,gt_P,gt_N = evaluation(infoMATpath,gt_img_path)
#         infoMATlist.append(dir+names)




def evaluation(imgx,num4indexX,imgy,num4indexY,path,truthimg_path,i = 0):



    maskNfreqID_infoMat = np.load(path)
    mask0 = maskNfreqID_infoMat[i].astype(bool)
    # mask25 = maskNfreqID_infoMat[2].astype(bool)
    # mask50 = maskNfreqID_infoMat[3].astype(bool)
    # mask75 = maskNfreqID_infoMat[4].astype(bool)




    gridwidth = int(imgx / num4indexX)
    gridheight = int(imgy / num4indexY)
    mask0 = maskNfreqID_infoMat[i]
    y_idx = np.nonzero(mask0)[0]
    x_idx = np.nonzero(mask0)[1]
    mask0 = np.zeros((imgy, imgx))
    for y, x in zip(y_idx, x_idx):
        mask0[y * gridheight:(y + 1) * gridheight, x * gridwidth:(x + 1) * gridwidth] = 1
    mask0 = mask0.astype(bool)







    truth_mask = cv2.imread(truthimg_path)[:, :, 0].astype(bool)
    Num_p = np.count_nonzero(truth_mask)
    Num_n = np.count_nonzero(1 - truth_mask)

    # cv2.imshow("truth_mask",truth_mask.astype(np.uint8)*255)
    # cv2.waitKey(0)

    TP_mask = np.logical_and(truth_mask, mask0)
    TN_mask = np.logical_and(1 - truth_mask, 1 - mask0)
    FP_mask = np.logical_and(1 - truth_mask, mask0)
    FN_mask = np.logical_and(truth_mask, 1 - mask0)

    # cv2.imshow("TP_mask", TP_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    # cv2.imshow("TN_mask", TN_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    # cv2.imshow("FP_mask", FP_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    # cv2.imshow("FN_mask", FN_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)

    TP = np.count_nonzero(TP_mask)
    FP = np.count_nonzero(FP_mask)
    TN = np.count_nonzero(TN_mask)
    FN = np.count_nonzero(FN_mask)
    pridic_P = TP+FP
    pridic_N = TN+FN

    precision =round( TP/(TP+FP),3 )if (TP+FP) != 0 else None#np.count_nonzero(TP_mask)/(np.count_nonzero(TP_mask)+np.count_nonzero(FP_mask))
    recall = round(TP/(TP+FN),3 )if (TP+FP) != 0 else None#np.count_nonzero(TP_mask)/(np.count_nonzero(TP_mask)+np.count_nonzero(FN_mask))
    F1 = round(2*precision*recall/(precision+recall),3 )if (precision!=None and precision+recall !=0)  else None
    # accurancy = (TP+TN)/(Num_n+Num_p)
    PA = round((TP+TN)/(Num_n+Num_p),3 )
    mPA =round((TP/Num_p+TN/Num_n)/2,3 )
    IoU_P = round(TP/(TP+FP+Num_p-TP),3 )
    IoU_N = round(TN/(TN+FN+Num_n-TN),3 )
    mIoU = round((IoU_N+IoU_P)/2,3 )
    FWIoU = round(Num_p/(Num_n+Num_p)*IoU_P+Num_n/(Num_n+Num_p)*IoU_N,3 )
    #TP, FP, TN, FN, gt_P, gt_N
    return (TP, FP, TN, FN,Num_p,Num_n,pridic_P,pridic_N,precision,recall,F1,PA,mPA,IoU_P,IoU_N,mIoU,FWIoU)




# for root, directories, filenames in os.walk(dir):
#     for directory in directories:
#         os.path.join(root, directory)
#     for filename in filenames:
#             print os.path.join(root,filename)


def writetoevaluation(imgx,num4indexX,imgy,num4indexY,gt_img_path,infoMatPath,eval_txt_path):
    # items = os.listdir(dir)
    # infoMATlist = []
    with open(eval_txt_path, 'a') as f:
        # f.write("0 2 3 4   represents the threshold value 0 0.25 0.5 0.75   ")
        # f.write('\n')
        # f.write(
        #     "order: TP, FP, TN, FN, groud_P, groud_N,pridict_P,pridict_N,precision,recall,F1,PA,mPA,IoU_P,IoU_N,mIoU,FWIoU\n")

        # for names in items:
        #     if names.endswith("ID.npy"):




                f.write(infoMatPath)
                f.write('\n')
                prefix = ["", "TP", "FP", "TN", "FN", "groud_P", "groud_N", "pridict_P", "pridict_N", "precision", "recall", "F1",
              "PA", "mPA", "IoU_P", "IoU_N", "mIoU", "FWIoU"]
                for j in prefix:
                    f.write(j.ljust(10))
                f.write('\n')
                # infoMATpath = dir + names
                for i in [0, 2, 3, 4]:
                    statistics = evaluation(imgx,num4indexX,imgy,num4indexY,infoMatPath, gt_img_path, i)  # 0 2 3 4
                    f.write(str(i) + ": ")
                    for j in statistics:
                        f.write(str(j).ljust(10))
                    f.write('\n')
                    # infoMATlist.append(dir + names)
                f.write('\n')
                f.close()

#
#
# gt_img_path = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6\\truth.png"
# dir = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6\GF\\ang\size512_384\\"
# items = os.listdir(dir)
# infoMATlist = []
# with open(dir+'evaluation.txt', 'a') as f:
#     f.write("0 2 3 4   represents the threshold value 0 0.25 0.5 0.75   ")
#     f.write('\n')
#     f.write("order: TP, FP, TN, FN, groud_P, groud_N,pridict_P,pridict_N,precision,recall,F1,PA,mPA,IoU_P,IoU_N,mIoU,FWIoU\n")
#
#     for names in items:
#         if names.endswith("ID.npy"):
#             f.write(names)
#             f.write('\n')
#             infoMATpath = dir + names
#             for i in [0,2,3,4]:
#                 statistics = evaluation(infoMATpath, gt_img_path, i)  # 0 2 3 4
#                 f.write(str(i)+": ")
#                 for j in statistics:
#                     f.write(str(j).ljust(10))
#                 f.write('\n')
#                 # infoMATlist.append(dir + names)
#             f.write('\n')
#     f.close()



