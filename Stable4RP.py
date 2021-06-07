# Import numpy and OpenCV
import numpy as np
import cv2
import time
# Read input video










import numpy as np
import cv2 as cv



def cumdot(A):
    B = np.empty(A.shape)
    B[0] = A[0]
    for i in range(1, A.shape[0]):
        # B[i] = B[i - 1] @ A[i]
        B[i] = np.dot(B[i - 1], A[i])
    return B
import matplotlib.pyplot as plt


#
#
#
# def get_matched_idx_pos_disparity(img1_path,img2_path):
#     img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # queryImage
#     img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # trainImage
#     orb = cv.ORB_create(nlevels=8)
#     kp1 = orb.detect(img1, None)
#     # compute the descriptors with ORB
#     kp1, des1 = orb.compute(img1, kp1)
#     kp2 = orb.detect(img2, None)
#     # compute the descriptors with ORB
#     kp2, des2 = orb.compute(img2, kp2)
#
#     # get the LUT of pos For 2 imgs
#     LUT_queryImg1 = []
#     LUT_trainImg2 = []
#     for i, n in enumerate(kp1):
#         LUT_queryImg1.append((i, n.pt))
#     LUT_queryImg1 = dict(LUT_queryImg1)
#     print('\nQUERY LUT', LUT_queryImg1)
#
#     for i, n in enumerate(kp2):
#         LUT_trainImg2.append((i, n.pt))
#     LUT_trainImg2 = dict(LUT_trainImg2)
#     print('\nTRAIN LUT', LUT_trainImg2)
#     LUT_queryImg1_des = []
#     LUT_trainImg2_des = []
#     for i in range(np.shape(des1)[0]):
#         LUT_queryImg1_des.append((i,list(des1[i])))
#     LUT_queryImg1_des = dict(LUT_queryImg1_des)
#     print('\nQUERY LUT of descriptor', LUT_queryImg1_des)
#
#     for i in range(np.shape(des2)[0]):
#         LUT_trainImg2_des.append((i,list(des2[i])))
#     LUT_trainImg2_des = dict(LUT_trainImg2_des)
#     print('\ntrain LUT of descriptor', LUT_trainImg2_des)
#
#
#     index_params = dict(algorithm=6,
#                         table_number=6,
#                         key_size=12,
#                         multi_probe_level=2)
#     search_params = {}
#     flann = cv.FlannBasedMatcher(index_params, search_params)
#
#     matches = flann.knnMatch(des1, des2, k=2)
#     # print('\nthe num of unfimaltered matched kp pairs :' + str(len(matches)))
#
#     # Need to draw only good matches, so create a mask
#     matchesMask = [[0, 0] for i in range(len(matches))]
#     # ratio test as per Lowe's paper
#     num_MP = 0
#     matched_idx = []
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.2 * n.distance:  # top two distances m n, a smaller parameter means a higher precision n less matching
#             matchesMask[i] = [1, 0]  # this mask encode connect the best m or the second best n
#             num_MP += 1
#             matched_idx.append([m.queryIdx, m.trainIdx])
#
#     matched_idxNpos = []
#     prvs_pts = []
#     next_pts = []
#
#     for i in matched_idx:
#         query_idx, train_idx = i[0], i[1]
#         prvs_pts.append( LUT_queryImg1[query_idx])
#         next_pts.append( LUT_trainImg2[train_idx])
#
#         matched_idxNpos.append([query_idx, LUT_queryImg1[query_idx], train_idx, LUT_trainImg2[train_idx]])
#
#
#     print('\nthe index N pos for matched kp (query,train):', matched_idxNpos)  # same   query is ordered.
#
#     draw_params = dict(matchColor=(255, 218, 185),
#                        singlePointColor=(255, 0, 0),
#                        matchesMask=matchesMask,
#                        flags=cv.DrawMatchesFlags_DEFAULT)
#     print('\nthe num of orb kp in query_img img1 :' + str(len(kp1)))
#     print('the num of orb kp in train_img img2 :' + str(len(kp2)))
#     print('the num of filtered matched kp pairs :' + str(num_MP))
#
#     img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
#     cv.namedWindow('', 0)  # so that the img will be clipped
#     cv.imshow('', img3)
#
#     cv.waitKey(0)
#     return prvs_pts,next_pts
# path1 =r'D:\Study\NCTcode\ImageSamples\left\\00002.png'
# path2 =r'D:\Study\NCTcode\ImageSamples\left\\00009.png'

# get_matched_idx_pos_disparity(path1,path2)



def transform(videopath):
    cap = cv2.VideoCapture(videopath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _, prev = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # （'I','4','2','0' 对应avi格式）
    path2 = videopath[:-4]+"_S"+".avi"
    out = cv2.VideoWriter( path2, fourcc, 25, (w,h))

    # Convert frame to grayscale
    img1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)
    # scaleX_list = np.zeros((n_frames - 1, 1), np.float32)
    # scaleY_list = np.zeros((n_frames - 1, 1), np.float32)
    pose_matrixs = np.zeros((n_frames - 1, 3, 3), np.float32)

    # transforms = []
    # scaleY_list = []
    # scaleX_list = []
    for j in range(n_frames - 2):
        success, img2 = cap.read()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if not success:
            break

        orb = cv.ORB_create(nlevels=8)
        kp1 = orb.detect(img1, None)
        # compute the descriptors with ORB
        kp1, des1 = orb.compute(img1, kp1)
        kp2 = orb.detect(img2, None)
        # compute the descriptors with ORB
        kp2, des2 = orb.compute(img2, kp2)

        # get the LUT of pos For 2 imgs
        LUT_queryImg1 = []
        LUT_trainImg2 = []
        for i, n in enumerate(kp1):
            LUT_queryImg1.append((i, n.pt))
        LUT_queryImg1 = dict(LUT_queryImg1)
        # print('\nQUERY LUT', LUT_queryImg1)

        for i, n in enumerate(kp2):
            LUT_trainImg2.append((i, n.pt))
        LUT_trainImg2 = dict(LUT_trainImg2)
        # print('\nTRAIN LUT', LUT_trainImg2)
        LUT_queryImg1_des = []
        LUT_trainImg2_des = []
        for i in range(np.shape(des1)[0]):
            LUT_queryImg1_des.append((i, list(des1[i])))
        LUT_queryImg1_des = dict(LUT_queryImg1_des)
        # print('\nQUERY LUT of descriptor', LUT_queryImg1_des)

        for i in range(np.shape(des2)[0]):
            LUT_trainImg2_des.append((i, list(des2[i])))
        LUT_trainImg2_des = dict(LUT_trainImg2_des)
        # print('\ntrain LUT of descriptor', LUT_trainImg2_des)

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        # print('\nthe num of unfimaltered matched kp pairs :' + str(len(matches)))

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        num_MP = 0
        matched_idx = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.2 * n.distance:  # top two distances m n, a smaller parameter means a higher precision n less matching
                matchesMask[i] = [1, 0]  # this mask encode connect the best m or the second best n
                num_MP += 1
                matched_idx.append([m.queryIdx, m.trainIdx])

        matched_idxNpos = []
        prvs_pts = np.zeros((len(matched_idx),1,2))
        next_pts = np.zeros((len(matched_idx),1,2))
        k = 0
        for i in matched_idx:
            query_idx, train_idx = i[0], i[1]
            prvs_pts[k] = np.array(LUT_queryImg1[query_idx])
            next_pts[k] = np.array(LUT_trainImg2[train_idx])
            k = k+1

        # print('\nthe index N pos for matched kp (query,train):', matched_idxNpos)  # same   query is ordered.

        draw_params = dict(matchColor=(255, 218, 185),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv.DrawMatchesFlags_DEFAULT)
        print("\n frame ID :",j)
        print('the num of orb kp in query_img img1 :' + str(len(kp1)))
        print('the num of orb kp in train_img img2 :' + str(len(kp2)))
        print('the num of filtered matched kp pairs :' + str(num_MP))

        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv.namedWindow('', 0)  # so that the img will be clipped
        cv.imshow('', img3)

        cv.waitKey(1)
        img1 = img2


        km = prvs_pts.astype(np.float32)
        km2 = next_pts.astype(np.float32)
        if len(km) < 3:
            print("Lost tracking! ")
            break
        # m = cv2.getAffineTransform(km, km2)
        m = cv2.estimateRigidTransform(km,km2,fullAffine=False)
        if  m is None  :
            print("transform is Nonetype!")
            break
        # dx = m[0, 2]
        # dy = m[1, 2]
        # da = np.arctan2(m[1, 0], m[0, 0])
        # scale_x = np.sqrt(m[0, 1] ** 2 + m[0, 0] ** 2)
        # scale_y = np.sqrt(m[1, 1] ** 2 + m[1, 0] ** 2)
        mat = np.zeros((3,3))
        mat[:2,:3] = m



        if j == 37:
            print()




        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[j] = [dx, dy, da]
        pose_matrixs[j] = mat.astype(np.float32)
    transforms = -np.cumsum(transforms,axis = 0)
    pose_matrixs = cumdot(pose_matrixs).astype(np.float32)







    # trajectory = np.cumsum(transforms, axis=0)
    # cum_scale_X = np.cumprod(scaleX_list, axis=0)
    # cum_scale_Y = np.cumprod(scaleY_list, axis=0)
    # transforms = -trajectory
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break






        dx = transforms[i, 0]
        dy = transforms[i, 1]
        da = transforms[i, 2]
        # if m is None:
        #     continue
        m = np.zeros((2,3))

        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy






        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_out = cv2.hconcat([frame, frame_stabilized])

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(0)
        # cv2.imwrite(stable_path + str(i) + '.png', frame_stabilized)
        out.write(frame_stabilized)




videopath = "D:\Study\Datasets\AEXTENSION\Cho80_extension\static_cam\pulseNstatic\\6.avi"

transform(videopath)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# cap = cv2.VideoCapture("D:\Study\Datasets\\moreCam.mp4")
# stable_path = "D:\Study\Datasets\\moreCam\\stable\\"
# # cap = cv2.VideoCapture("D:\Study\Datasets\\moreCamBest.mp4")
# # stable_path = "D:\Study\Datasets\\moreCamBestNontrembling\\set\\"
# # cap = cv2.VideoCapture(0)
#
# # Get frame count
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# n_frames  =50
#
# # Get width and height of video stream
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # Define the codec for output video
# # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # （'I','4','2','0' 对应avi格式）
# # fps = 30
#
# # Set up output video
# # out = cv2.VideoWriter("D:\Study\Datasets\\Stabledmorecam.avi", fourcc, fps, (w,h))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Read first frame
# _, prev = cap.read()
#
# # Convert frame to grayscale
# prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#
# # Pre-define transformation-store array
# transforms = np.zeros((n_frames - 1, 3), np.float32)
# scaleX_list = np.zeros((n_frames - 1, 1), np.float32)
# scaleY_list = np.zeros((n_frames - 1, 1), np.float32)
#
# m_list = []
#
# for i in range(n_frames - 2):
#     # Detect feature points in previous frame
#     prev_pts = cv2.goodFeaturesToTrack(prev_gray,
#                                        maxCorners=200,
#                                        qualityLevel=0.5,
#                                        minDistance=5,
#                                        blockSize=2)
#
#     # Read next frame
#     success, curr = cap.read()
#     if not success:
#         break
#
#         # Convert to grayscale
#     curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
#
#     # Calculate optical flow (i.e. track feature points)
#     curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
#
#     # Sanity check
#     assert prev_pts.shape == curr_pts.shape
#
#     # Filter only valid points
#     idx = np.where(status == 1)[0]
#     if status.all() == 0:
#         idx = np.where(status == (0))[0]
#
#     prev_pts = prev_pts[idx]
#     curr_pts = curr_pts[idx]
#
#
#     if i == 72:
#         print("jinjing")
#
#     # Find transformation matrix
#     #warp_mat = cv.getAffineTransform(srcTri, dstTri)
#     # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less
#     m = cv2.getAffineTransform(prev_pts[2:5], curr_pts[2:5])
#     m_list.append(m)
#     if m is None: #jinjing
#         m = m_last
#         # Extract traslation
#     dx = m[0, 2]
#     dy = m[1, 2]
#
#         # Extract rotation angle
#     da = np.arctan2(m[1, 0], m[0, 0])
#     scale_x =np.sqrt(m[0, 1]**2+ m[0, 0]**2)
#     scale_y =np.sqrt(m[1, 1]**2+ m[1, 0]**2)
#
#
#
#
#
#     m_last = m.copy()#jinjing
#         # Store transformation
#     transforms[i] = [dx, dy, da]
#     scaleX_list[i] = scale_x
#     scaleY_list[i] = scale_y
#
#
#         # Move to next frame
#     prev_gray = curr_gray
#
#     print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
#
#
# # Compute trajectory using cumulative sum of transformationstrajectory = np.cumsum(transforms, axis=0
#
# def movingAverage(curve, radius):
#       window_size = 2 * radius + 1
#       # Define the filter
#       f = np.ones(window_size)/window_size
#       # Add padding to the boundaries
#       curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
#       # Apply convolution
#       curve_smoothed = np.convolve(curve_pad, f, mode='same')
#       # Remove padding
#       curve_smoothed = curve_smoothed[radius:-radius]
#       # return smoothed curve
#       return curve_smoothed
#
#
# def smooth(trajectory):
#     smoothed_trajectory = np.copy(trajectory)
#     # Filter the x, y and angle curves
#     for i in range(3):
#         smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=5)
#
#     return smoothed_trajectory
#
#
#
# def fixBorder(frame):
#         s = frame.shape
#         # Scale the image 4% without moving the center
#         T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
#         frame = cv2.warpAffine(frame, T, (s[1], s[0]))
#         return frame
#
#
# # Compute trajectory using cumulative sum of transformations
# trajectory = np.cumsum(transforms, axis=0)
# cum_scale_X = np.cumprod(scaleX_list,axis = 0)
# cum_scale_Y = np.cumprod(scaleY_list,axis = 0)
#
# # Calculate difference in smoothed_trajectory and trajectory
# difference = smooth(trajectory) - trajectory
#
# # Calculate newer transformation array
# # transforms_smooth = transforms + difference    #  换成-traj  (注意不是-trajectory) 可以完全消除运动
# transforms_smooth = -trajectory
#
# # Reset stream to first frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# # Write n_frames-1 transformed frames
# for i in range(n_frames - 2):
#     # Read next frame
#     success, frame = cap.read()
#     if not success:
#         break
#
#     # Extract transformations from the new transformation array
#     dx = transforms_smooth[i, 0]
#     dy = transforms_smooth[i, 1]
#     da = transforms_smooth[i, 2]
#
#     # Reconstruct transformation matrix accordingly to new values
#     m = np.zeros((2, 3), np.float32)
#     scaleX = 1.0/cum_scale_X[i,0]
#     scaleY = 1.0 / cum_scale_Y[i, 0]
#     m[0, 0] = np.cos(da)*scaleX
#     m[0, 1] = -np.sin(da)*scaleX
#     m[1, 0] = np.sin(da)*scaleY
#     m[1, 1] = np.cos(da)*scaleY
#     m[0, 2] = dx
#     m[1, 2] = dy
#
#     # Apply affine wrapping to the given frame
#     frame_stabilized = cv2.warpAffine(frame, m, (w, h))
#
#
#
# #        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
# #         frame = cv2.warpAffine(frame, T, (s[1], s[0]))
#     # Fix border artifacts
#     # frame_stabilized = fixBorder(frame_stabilized)
#
#     # Write the frame to the file
#     frame_out = cv2.hconcat([frame, frame_stabilized])
#
#     # If the image is too big, resize it.
#     # if (frame_out.shape[1]):
#     #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));
#
#     cv2.imshow("Before and After", frame_out)
#     cv2.waitKey(0)
#     cv2.imwrite(stable_path + str(i) + '.png', frame_stabilized)
#     # out.write(frame_out)
#
#
#
