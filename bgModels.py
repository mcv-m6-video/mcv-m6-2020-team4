import numpy as np
import cv2
from data import number_of_images_jpg
import imageio
import copy

def unsqueeze(img):

    img = np.expand_dims(img, -1)

    return img


def bg_model(frames_path, color_space=cv2.COLOR_BGR2GRAY):
    """
    This function reads the 25% of the frames and calculates the initial
    gaussian model for each pixel.
    Returns two matrices with the mean and the variance for each pixel.
    It takes quite a long time...
    """



    video_n_frames = number_of_images_jpg(frames_path)
    p25_frames = int(video_n_frames*0.25)
    img = cv2.imread(frames_path+'/frame_0001.jpg')
    img = cv2.cvtColor(img, color_space)

    img = unsqueeze(img)

    imga = np.zeros((p25_frames, *img.shape)).astype(np.float32)
    print('Reading frames ')
    for i in range(0, p25_frames):

        img = cv2.imread(frames_path+('/frame_{:04d}.jpg'.format(i+1)))
        imga[i, ...] = unsqueeze(cv2.cvtColor(img, color_space).astype(np.float32))

    # mean
    print('Calculating mean .... (takes a while)')
    mu = np.mean(imga, axis = (0, -1), dtype = np.float32)
    # variance
    print('Calculating variance .... (takes a while)')
    sigma = np.std(imga, axis = (0, -1), dtype = np.float32)

    print('End')
    return mu, sigma


def remove_bg(mu, sigma, alpha, frames_path, initial_frame, final_frame,
              animation = False, denoise = False, color_space=cv2.COLOR_BGR2GRAY):
    """
    Save detected bb in the same format as GT which is:
        'frame', 'label', 'id', 'xtl','ytl','xbr','ybr'
    """
    c = 0
    detected_bb = []
    for i in range(initial_frame, final_frame):
        #read image
        img = cv2.imread(frames_path+('/frame_{:04d}.jpg'.format(i+1)))
        img = cv2.cvtColor(img, color_space).astype(np.float32)


        if i == initial_frame and animation:
            sx, sy = np.int(np.shape(img)[0]/4), np.int(np.shape(img)[1]/4)
            frames = np.zeros((final_frame-initial_frame, sx, sy)).astype(np.uint8())

        frame = np.zeros(np.shape(img)).astype(np.uint8())
        frame[np.abs(img-mu)>= alpha*(sigma+2)] = 255
        frame[np.abs(img-mu)< alpha*(sigma+2)] = 0
        if animation:
            frames[c,...] = cv2.resize(frame, (sy,sx))

        if denoise:
            frame = cv2.medianBlur(frame,7)

        (_,contours,_) = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # Open cv documentation:  (x,y) be the top-left coordinate
            #of the rectangle and (w,h) be its width and height.
            (x,y,w,h) = cv2.boundingRect(contour)
            if w>10 and h>10:
                detected_bb.append([i, 'car', 0, x, y, x+w, y+h])
        c = c+1
    if animation:
        imageio.mimsave('bg_removal_a{}.gif'.format(alpha), frames)

    return detected_bb

def remove_adaptive_bg(mu_original, sigma_original, alpha, rho, frames_path,
                       initial_frame, final_frame, animation = False, denoise = False, color_space=cv2.COLOR_BGR2GRAY):
    """
    Save detected bb in the same format as GT which is:
        'frame', 'label', 'id', 'xtl','ytl','xbr','ybr'
    """
    mu = copy.deepcopy(mu_original)
    sigma = copy.deepcopy(sigma_original)

    c = 0
    detected_bb = []
    for i in range(initial_frame, final_frame):
        #read image
        img = cv2.imread(frames_path+('/frame_{:04d}.jpg'.format(i+1)))
        img = cv2.cvtColor(img, color_space).astype(np.float32)

        if i == initial_frame and animation:
            sx, sy = np.int(np.shape(img)[0]/4), np.int(np.shape(img)[1]/4)
            frames = np.zeros((final_frame-initial_frame, sx, sy)).astype(np.uint8())

        frame = np.zeros(np.shape(img)).astype(np.uint8())
        frame[np.abs(img-mu)>= alpha*(sigma+2)] = 255
        frame[np.abs(img-mu)< alpha*(sigma+2)] = 0

        #Update mu and sigma if needed
        mu[frame == 0] = rho * img[frame == 0] + (1 - rho) * mu[frame == 0]
        sigma[frame == 0] = np.sqrt(rho * np.power((img[frame == 0] - mu[frame == 0]), 2) + (1 - rho) * np.power(sigma[frame == 0], 2))

        if len(frame.shape) != 2:
            frame = np.ascontiguousarray(frame[..., 0])

        if animation:
            frames[c,...] = cv2.resize(frame, (sy,sx))

        if denoise:
            frame = cv2.medianBlur(frame,7)

        (_,contours,_) = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # Open cv documentation:  (x,y) be the top-left coordinate
            #of the rectangle and (w,h) be its width and height.
            (x,y,w,h) = cv2.boundingRect(contour)
            if w>10 and h>10:
                detected_bb.append([i, 'car', 0, x, y, x+w, y+h])
        c = c+1
    if animation:
        imageio.mimsave('bg_removal_a{}_p{}_{}.gif'.format(alpha, rho, color_space), frames)

    return detected_bb
