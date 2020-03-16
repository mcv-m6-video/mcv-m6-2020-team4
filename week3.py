import cv2
import matplotlib.pylab as plt
import numpy as np
import imageio

from data import read_detections_file
from data import save_frames, number_of_images_jpg
from metrics.mAP import calculate_ap
from utils.visualization import animation_tracks
from tracking import tracking_iou, kalman_filter_tracking


def main():
    frames_path = 'datasets/AICity_data/train/S03/c010/data'
    save_frames(frames_path)
    print("Finished saving")

    print("Task 2.1")
    #task21()
    print("Task 2.2")
    model_type = 1 # Constant acceleration
    task22("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt", frames_path, model_type)

def task21():
    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
    #det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_ssd512.txt")
    #det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_yolo3.txt")


    video_n_frames = number_of_images_jpg(frames_path)

    det_bb, idd = tracking_iou(det_bb, video_n_frames)
    ini_frame = 550
    end_frame = 650

    animation_tracks(det_bb, idd, ini_frame, end_frame, frames_path)

def task22(detections_file, frames_path, model_type):
    det_bb = read_detections_file(detections_file)
    #TODO: filter detections (class + confidence)
    video_n_frames = number_of_images_jpg(frames_path)
    det_bb = kalman_filter_tracking(det_bb, video_n_frames, model_type)

    max_id = max(det_bb, key=lambda x: x[2])[2]
    ini_frame = 550
    end_frame = 650
    animation_tracks(det_bb, max_id, ini_frame, end_frame, frames_path)


if __name__ == '__main__':
    main()
