import numpy as np
import cv2
import os

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



