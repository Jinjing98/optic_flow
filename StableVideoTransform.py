# Import numpy and OpenCV
import numpy as np
import cv2
import time
# Read input video
cap = cv2.VideoCapture("D:\Study\Datasets\\moreCam.mp4")
stable_path = "D:\Study\Datasets\\moreCam\\stable\\"
# cap = cv2.VideoCapture("D:\Study\Datasets\\moreCamBest.mp4")
# stable_path = "D:\Study\Datasets\\moreCamBestNontrembling\\set\\"
# cap = cv2.VideoCapture(0)

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # （'I','4','2','0' 对应avi格式）
# fps = 30

# Set up output video
# out = cv2.VideoWriter("D:\Study\Datasets\\Stabledmorecam.avi", fourcc, fps, (w,h))















# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((n_frames - 1, 3), np.float32)

for i in range(n_frames - 2):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.5,
                                       minDistance=5,
                                       blockSize=2)

    # Read next frame
    success, curr = cap.read()
    if not success:
        break

        # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    if status.all() == 0:
        idx = np.where(status == (0))[0]

    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]


    if i == 631:
        print("jinjing")

    # Find transformation matrix
    #warp_mat = cv.getAffineTransform(srcTri, dstTri)
    m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less
    # m = cv2.getAffineTransform(prev_pts, curr_pts)
    if m is None: #jinjing
        m = m_last
        # Extract traslation
    dx = m[0, 2]
    dy = m[1, 2]

        # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])





    m_last = m.copy()#jinjing
        # Store transformation
    transforms[i] = [dx, dy, da]

        # Move to next frame
    prev_gray = curr_gray

    print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))


# Compute trajectory using cumulative sum of transformationstrajectory = np.cumsum(transforms, axis=0

def movingAverage(curve, radius):
      window_size = 2 * radius + 1
      # Define the filter
      f = np.ones(window_size)/window_size
      # Add padding to the boundaries
      curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
      # Apply convolution
      curve_smoothed = np.convolve(curve_pad, f, mode='same')
      # Remove padding
      curve_smoothed = curve_smoothed[radius:-radius]
      # return smoothed curve
      return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=5)

    return smoothed_trajectory



def fixBorder(frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame


# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

# Calculate difference in smoothed_trajectory and trajectory
difference = smooth(trajectory) - trajectory

# Calculate newer transformation array
# transforms_smooth = transforms + difference    #  换成-traj  (注意不是-trajectory) 可以完全消除运动
transforms_smooth = -trajectory

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(n_frames - 2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))



#        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
#         frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    # Fix border artifacts
    # frame_stabilized = fixBorder(frame_stabilized)

    # Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])

    # If the image is too big, resize it.
    # if (frame_out.shape[1]):
    #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)
    cv2.imwrite(stable_path + str(i) + '.png', frame_stabilized)
    # out.write(frame_out)



