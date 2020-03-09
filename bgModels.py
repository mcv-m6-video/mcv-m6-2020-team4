import copy

import cv2
import imageio
import numpy as np

from data import number_of_images_jpg


def bg_model_grayscale(frames_path):
    """
    This function reads the 25% of the frames and calculates the initial
    gaussian model for each pixel.
    Returns two matrices with the mean and the variance for each pixel.
    It takes quite a long time...
    """
    video_n_frames = number_of_images_jpg(frames_path)
    p25_frames = int(video_n_frames * 0.25)
    img = cv2.imread(frames_path + '/frame_0001.jpeg', 0)
    imga = np.zeros((p25_frames, np.shape(img)[0], np.shape(img)[1])).astype(np.float16())
    print('Reading frames ')
    for i in range(0, p25_frames):
        # in openCV imread(path,0) the 0 is already grayscale!
        imga[i, ...] = cv2.imread(frames_path + ('/frame_{:04d}.jpeg'.format(i + 1)), 0).astype(np.float16())

    # mean
    print('Calculating mean .... (takes a while)')
    mu = np.mean(imga, axis=0, dtype=np.float64)
    # variance
    print('Calculating variance .... (takes a while)')
    sigma = np.std(imga, axis=0, dtype=np.float64)

    print('End')
    return mu, sigma


def remove_bg(mu, sigma, alpha, frames_path, initial_frame, final_frame,
              animation=False, denoise=False):
    """
    Save detected bb in the same format as GT which is:
        'frame', 'label', 'id', 'xtl','ytl','xbr','ybr'
    """
    c = 0
    detected_bb = []
    for i in range(initial_frame, final_frame):
        # read image
        img = cv2.imread(frames_path + ('/frame_{:04d}.jpeg'.format(i + 1)), 0).astype(np.float64())

        if i == initial_frame and animation:
            sx, sy = np.int(np.shape(img)[0] / 4), np.int(np.shape(img)[1] / 4)
            frames = np.zeros((final_frame - initial_frame, sx, sy)).astype(np.uint8())

        frame = np.zeros(np.shape(img)).astype(np.uint8())
        frame[np.abs(img - mu) >= alpha * (sigma + 2)] = 255
        frame[np.abs(img - mu) < alpha * (sigma + 2)] = 0
        if animation:
            frames[c, ...] = cv2.resize(frame, (sy, sx))

        if denoise:
            frame = cv2.medianBlur(frame, 7)

        (_, contours, _) = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # Open cv documentation:  (x,y) be the top-left coordinate
            # of the rectangle and (w,h) be its width and height.
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                detected_bb.append([i, 'car', 0, x, y, x + w, y + h])
        c = c + 1
    if animation:
        imageio.mimsave('bg_removal_a{}.gif'.format(alpha), frames)

    return detected_bb


def remove_adaptive_bg(mu_original, sigma_original, alpha, rho, frames_path,
                       initial_frame, final_frame, animation=False, denoise=False):
    """
    Save detected bb in the same format as GT which is:
        'frame', 'label', 'id', 'xtl','ytl','xbr','ybr'
    """
    mu = copy.deepcopy(mu_original)
    sigma = copy.deepcopy(sigma_original)
    c = 0
    detected_bb = []
    for i in range(initial_frame, final_frame):
        # read image
        img = cv2.imread(frames_path + ('/frame_{:04d}.jpg'.format(i + 1)), 0).astype(np.float64())

        if i == initial_frame and animation:
            sx, sy = np.int(np.shape(img)[0] / 4), np.int(np.shape(img)[1] / 4)
            frames = np.zeros((final_frame - initial_frame, sx, sy)).astype(np.uint8())

        frame = np.zeros(np.shape(img)).astype(np.uint8())
        frame[np.abs(img - mu) >= alpha * (sigma + 2)] = 255
        frame[np.abs(img - mu) < alpha * (sigma + 2)] = 0

        # Update mu and sigma if needed
        mu[frame == 0] = rho * img[frame == 0] + (1 - rho) * mu[frame == 0]
        sigma[frame == 0] = np.sqrt(
            rho * np.power((img[frame == 0] - mu[frame == 0]), 2) + (1 - rho) * np.power(sigma[frame == 0], 2))

        if animation:
            frames[c, ...] = cv2.resize(frame, (sy, sx))

        if denoise:
            frame = cv2.medianBlur(frame, 7)

        (_, contours, _) = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # Open cv documentation:  (x,y) be the top-left coordinate
            # of the rectangle and (w,h) be its width and height.
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                detected_bb.append([i, 'car', 0, x, y, x + w, y + h])
        c = c + 1
    if animation:
        imageio.mimsave('bg_removal_a{}_p{}.gif'.format(alpha, rho), frames)

    return detected_bb


def bg_estimation(mode, **kwargs):
    if mode == 'mog':
        return cv2.createBackgroundSubtractorMOG2(varThreshold=kwargs['varThreshold'], detectShadows=True)
    elif mode == 'knn':
        return cv2.createBackgroundSubtractorKNN(dist2Threshold=kwargs['dist2Threshold'], detectShadows=True)
    elif mode == 'gmg':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=kwargs['initializationFrames'])
