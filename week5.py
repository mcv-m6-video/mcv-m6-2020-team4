import os
import cv2
import copy

from utils.data import read_detections_file, filter_gt, read_gt_txt, filter_det_confidence, number_of_images_jpg
from tracking.new_tracking import tracking_iou
from tracking.tracking import kalman_filter_tracking
from metrics.mAP import calculate_ap
from utils.postprocessing import clean_tracks, remove_parked
from opt import parse_args_week5
from utils.visualization import animation_tracks, animation_2bb

def main():
    opt = parse_args_week5()
    print(opt.__dict__)

    if opt.task == 1:
        print("Starting task 1")
        task1(opt.detector, opt.trackingMethod, opt.postprocessing, opt.visualization)

    elif opt.task == 2:
        print("Starting task2")

def task1(detector, tracking_method, postprocessing, visualization):
    #Read gt
    gt_annot_file = 'datasets/AICity_data/train/S03/c010/gt/gt.txt'
    gt_bb = read_gt_txt(gt_annot_file)

    #Read and filter detections
    if detector == "MaskR-CNN":
        detections_file = "datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
    elif detector == "YOLO":
        detections_file = "datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"
    elif detector == "SSD":
        detections_file = "datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"
    det_bb = read_detections_file(detections_file)
    det_bb = filter_det_confidence(det_bb, threshold = 0.5)
    print("Detections loaded")

    #Tracking
    frames_path = 'datasets/AICity_data/train/S03/c010/data'
    video_n_frames = number_of_images_jpg(frames_path)
    if tracking_method == "MaxOverlap":
        det_bb_tracking, id_max = tracking_iou(frames_path, copy.deepcopy(det_bb), video_n_frames, mode='other')
    elif tracking_method == "Kalman":
        det_bb_tracking = kalman_filter_tracking(copy.deepcopy(det_bb), video_n_frames, 0)
        id_max = max([v[2] for v in  det_bb_tracking])
    print("Tracking finished")

    #Postprocessing
    det_bb_clean = clean_tracks(copy.deepcopy(det_bb_tracking), id_max)
    if postprocessing == "RemoveParked":
        det_bb_clean = remove_parked(copy.deepcopy(det_bb_clean), id_max, threshold = 10.0)
    print("Postprocessing finished")
    #Results
    ap = calculate_ap(det_bb_clean, copy.deepcopy(gt_bb), 0, video_n_frames, mode='sort')
    print("Average precision: {}".format(ap))

    #Create animation if required
    if visualization:
        ini_frame = 550
        end_frame = 650
        frames_path = 'datasets/AICity_data/train/S03/c010/data'
        animation_tracks(det_bb_clean, id_max, ini_frame, end_frame, frames_path)
        print("Animation stored")

if __name__ == '__main__':
    main()
