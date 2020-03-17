import cv2
import matplotlib.pylab as plt
import numpy as np
import imageio
import copy

from data import read_detections_file, read_xml_gt_options
from data import save_frames, number_of_images_jpg, filter_gt
from metrics.mAP import calculate_ap
from utils.visualization import animation_tracks
from tracking import tracking_iou, kalman_filter_tracking





def main():
    frames_path = 'datasets/AICity_data/train/S03/c010/data'
    gt_path = "datasets/ai_challenge_s03_c010-full_annotation.xml"
    save_frames(frames_path)
    print("Finished saving")

    print("Task 2.1")
    task21(frames_path, gt_path)
    print("Task 2.2")
#    model_type = 1 # Constant acceleration
#    task22("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt", frames_path, model_type)

def task21(frames_path, gt_path):
    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
    #det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_ssd512.txt")
    #det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_yolo3.txt")
    gt_bb = read_xml_gt_options(gt_path, False, False)
    gt_bb = filter_gt(gt_bb, ['car'])
    
    video_n_frames = number_of_images_jpg(frames_path)

    det_bb1, idd = tracking_iou(copy.deepcopy(det_bb), video_n_frames)

    ap = calculate_ap(det_bb1, gt_bb, 0, video_n_frames, mode='area')
    print('AP maximum iou track: ', ap)
    ini_frame = 420
    end_frame = 580
    animation_tracks(det_bb1, idd, ini_frame, end_frame, frames_path)


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
