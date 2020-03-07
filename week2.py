

import numpy as np
import matplotlib.pylab as plt

from data import read_xml_gt, FrameCapture
from data import save_frames, number_of_frames
from utils.visualization import animate_iou, animation_2bb, plot_animation
from bgModels import bg_model_grayscale, remove_bg



path = 'datasets/AICity_data/train/S03/c010'        
save_frames(path)

video_path = 'datasets/AICity_data/train/S03/c010/vdo.avi'
frames_path = 'datasets/AICity_data/train/S03/c010/data'
mu, sigma = bg_model_grayscale(video_path, frames_path)


video_n_frames = number_of_frames(video_path)
detections = remove_bg(mu, sigma, 3, int(video_n_frames*0.25) + 1, int(video_n_frames*0.25) + 101, 
                       animation = False)


gt_bb = read_xml_gt("datasets/ai_challenge_s03_c010-full_annotation.xml")
animation_2bb('try', '.gif', gt_bb, detections, frames_path, 10, 10, int(video_n_frames*0.25) + 1, 
              int(1920 / 2), int(1080 / 2))
    
