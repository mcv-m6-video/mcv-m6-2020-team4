from glob import glob
import os
import cv2
import numpy as np


def get_files_from_dir(path, extension):
    assert os.path.exists(path), "Path not found"
    return sorted(glob(os.path.join(path, "*.{}".format(extension))))


def fix_border(frame):
    s = frame.shape
    m = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, m, (s[1], s[0]))

    return frame


def moving_average(curve, radius):
    window_size = 2 * radius + 1

    # Define the filter
    f = np.ones(window_size) / window_size

    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')

    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')

    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth(trajectory, radius):

    smoothed_trajectory = np.copy(trajectory)

    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius)

    return smoothed_trajectory


# Borrowed from https://github.com/francocurotto/Video-Stabilization
def getVideoArray(videoPath):
    # video in info
    video = cv2.VideoCapture(videoPath)
    N_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(video.get(cv2.CAP_PROP_FPS))
    VID_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    VID_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # numpy array
    videoArr = np.zeros((N_FRAMES, VID_HEIGHT, VID_WIDTH), dtype=np.uint8)
    # fill array
    for i in range(N_FRAMES):
        _, videoArr[i, :, :] = readVideoGray(video)
    video.release()
    return videoArr


def readVideoGray(video):
    ret, frame = video.read()
    if ret:
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frameGray = None
    return ret, frameGray


def getTrans(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt, fast=True):

    N_FRAMES = videoArr.shape[0]
    trans = np.zeros((N_FRAMES, 3, 3))

    localMotion = getLocalMotionFast(
        videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES)

    for i in range(N_FRAMES):
        for x in range(3):
            for y in range(3):
                trans[i, x, y] = np.dot(filt, localMotion[i, :, x, y])

    return trans


def getLocalMotionFast(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES):
    N_FRAMES = videoArr.shape[0]
    FILT_WIDTH = filt.size
    halfFilt = FILT_WIDTH // 2
    localMotion = np.zeros((N_FRAMES, FILT_WIDTH, 3, 3))

    # get next frame motion with ORB (and same frame with identity)
    for i in range(N_FRAMES):
        localMotion[i, halfFilt, :, :] = np.identity(3)
        try:
            localMotion[i, halfFilt + 1, :, :] = estMotion(videoArr[i, :, :], videoArr[i + 1, :, :], detector, bf, MATCH_THRES, RANSAC_THRES, show=False)
        except IndexError:
            localMotion[i, halfFilt + 1, :, :] = np.identity(3)

    # get n-step frame motion from next step motion
    for j in range(halfFilt + 2, FILT_WIDTH):
        for i in range(N_FRAMES):
            try:
                localMotion[i, j, :, :] = np.dot(
                    localMotion[i + 1, j - 1, :, :], localMotion[i, j - 1, :, :])
            except IndexError:
                localMotion[i, j, :, :] = np.identity(3)

    # get past n-step motion (by inversion of forward motion)
    for j in range(halfFilt):
        for i in range(N_FRAMES):
            try:
                localMotion[i, j, :, :] = np.linalg.inv(
                    localMotion[i + j - halfFilt, FILT_WIDTH - j - 1, :, :])
            except IndexError:
                localMotion[i, j, :, :] = np.identity(3)

    return localMotion


def estMotion (frame1, frame2, detector, bf, MATCH_THRES, RANSAC_THRES, show=False):

    try:
        # get keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(frame1, None)
        kp2, des2 = detector.detectAndCompute(frame2, None)

        # get matches
        matches = bf.match(des1, des2)
        matches = filterMatches(matches, MATCH_THRES)

        # get affine transform
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #M, mask = cv2.findHomography(src_pts, dst_pts, 0)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRES)
        #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

        #plotMatches(frame1, kp1, frame2, kp2, matches, 0)
        matches = maskMatches(matches, mask)

        if show:
            plotMatches(frame1, kp1, frame2, kp2, matches, 0)
    except:
        M = np.identity(3)

    return M

def reconVideo (videoInPath, videoOutPath, trans, BORDER_CUT):

    # video in info
    videoIn = cv2.VideoCapture(videoInPath)
    N_FRAMES = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(videoIn.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    VID_WIDTH = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    VID_HEIGHT = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # video out creation
    videoInSize = (int(VID_WIDTH), int(VID_HEIGHT))
    videoOutSize = (int(VID_WIDTH) - 2*BORDER_CUT, int(VID_HEIGHT) - 2*BORDER_CUT)
    videoOut = cv2.VideoWriter(videoOutPath, fourcc, FPS, videoOutSize)

    # frame transformation
    for i in range(N_FRAMES):
        ret, frame = videoIn.read()
        frameOut = cv2.warpPerspective(frame, trans[i,:,:], videoInSize, flags=cv2.INTER_NEAREST)
        frameOut = frameOut[BORDER_CUT:-BORDER_CUT, BORDER_CUT:-BORDER_CUT]
        videoOut.write(frameOut)

    videoIn.release()
    videoOut.release()
