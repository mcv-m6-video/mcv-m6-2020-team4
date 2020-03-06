import cv2
import os
import numpy as np
import matplotlib.pylab as plt
import imageio



# Function to extract frames 
def FrameCapture(path,directory): 
    """
    To reconstruct the video with the bounding boxes we need to extract the frames
    and save them somewhere
    """
    # Path to video file 
    vidObj = cv2.VideoCapture(path + "/vdo.avi") 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    if os.path.isfile(directory + "/frame_0000.jpg"):
        pass
    else:
        while success: 
      
            # vidObj object calls read 
            # function extract frames 
            success, image = vidObj.read() 
      
            # Saves the frames with frame-count 
            cv2.imwrite(directory + "/frame_{:04d}.jpg".format(count), image) 
      
            count += 1
            print('frame saved: ', count)

def save_frames(path):
    """
    This will automatically create a floder containing the frames of the video
    If it already exists then it won't save the frames again.
    """
    directory = path + '/data'
    if not os.path.exists(directory):
        os.makedirs(directory)        
    FrameCapture(path, directory)
    
   
def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

    
    
def bg_model_grayscale(video_path, frames_path):
    """
    This function reads the 25% of the frames and calculates the initial
    gaussian model for each pixel.
    Returns two matrices with the mean and the variance for each pixel.
    It takes quite a long time...
    """
    video_n_frames = number_of_frames(video_path)
    p25_frames = int(video_n_frames*0.25)
    img = cv2.imread(frames_path+'/frame_0000.jpg',0)
    imga = np.zeros((p25_frames, np.shape(img)[0], np.shape(img)[1])).astype(np.float32())
    for i in range(0, p25_frames):
        print(i)
        imga[i,...] = cv2.imread(frames_path+('/frame_{:04d}.jpg'.format(i)),0).astype(np.float32())
        
    # mean
    mu = np.mean(imga, axis = 0)         
    # variance
    sigma = np.std(imga, axis = 0)    
    
    return mu, sigma

    
def remove_bg(mu, sigma, alpha, initial_frame, final_frame, animation = False):
    c = 0
    for i in range(initial_frame, final_frame):
        img = cv2.imread(frames_path+('/frame_{:04d}.jpg'.format(i)),0).astype(np.float32())
        if i == initial_frame:
            frames = np.zeros((final_frame-initial_frame, np.shape(img)[0], np.shape(img)[1])).astype(np.uint8())
        frame = np.zeros(np.shape(img)).astype(np.uint8())
        frame[(img-mu)>= alpha*(sigma+2)] = 255
        frame[(img-mu)< alpha*(sigma+2)] = 0
        frames[c,...] = frame
        c = c+1
    if animation:
        imageio.mimsave('bg_removal_a{}.gif'.format(alpha), frames)
    return frames
    

#x,y,w,h = cv.boundingRect(cnt)
#cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


path = 'datasets/AICity_data/train/S03/c010'        
save_frames(path)

video_path = 'datasets/AICity_data/train/S03/c010/vdo.avi'
frames_path = 'datasets/AICity_data/train/S03/c010/data'
mu, sigma = bg_model_grayscale(video_path, frames_path)


video_n_frames = number_of_frames(video_path)
frames = remove_bg(mu, sigma, 1, int(video_n_frames*0.25) + 1, int(video_n_frames*0.25) + 101, 
                   animation = True)
