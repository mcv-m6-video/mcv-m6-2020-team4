import cv2
import matplotlib.pylab as plt
import numpy as np
import imageio

from bgModels import bg_model, remove_bg, bg_estimation, fg_segmentation_to_boxes, denoise_bg
from data import read_xml_gt_options, read_detections_file
from data import save_frames, number_of_images_jpg, filter_gt
from metrics.mAP import calculate_ap
from metrics.iou import bbox_iou
from opt import parse_args
from utils.visualization import animation_2bb, frame_with_2bb
from glob import glob
from utils.utils import get_files_from_dir
from utils.visualization import animation_2bb, frames_to_gif, animation_tracks
from tracking import tracking_iou


frames_path = 'datasets/AICity_data/train/S03/c010/data'
save_frames(frames_path)
print("Finished saving")



"TASK 2.1"
det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
#det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_ssd512.txt")
#det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_yolo3.txt")


video_n_frames = number_of_images_jpg(frames_path)

det_bb, idd = tracking_iou(det_bb, video_n_frames)
ini_frame = 550
end_frame = 650

animation_tracks(det_bb, idd, ini_frame, end_frame, frames_path)
