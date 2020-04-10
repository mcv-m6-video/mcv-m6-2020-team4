import os
import cv2
import copy

from utils.data import read_detections_file, filter_gt, read_xml_gt_options, filter_det_confidence, number_of_images_jpg
from tracking.new_tracking import tracking_iou
from tracking.tracking import kalman_filter_tracking
from metrics.mAP import calculate_ap
from opt import parse_args_week5

def main():
    opt = parse_args_week5()
    print(opt.__dict__)

    if opt.task == 1:
        print("Starting task 1")
        task1(opt.detector, opt.trackingMethod, opt.postprocessing)

    elif opt.task == 2:
        print("Starting task2")

def task1(detector, tracking_method, postprocessing):
    #Read and filter gt
    gt_annot_file = 'datasets/ai_challenge_s03_c010-full_annotation.xml'
    gt_bb = read_xml_gt_options(gt_annot_file, False, False)
    gt_bb = filter_gt(gt_bb, ["car"])

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
        det_bb_tracking, idd = tracking_iou(frames_path, copy.deepcopy(det_bb), video_n_frames, mode='other')
    elif tracking_method == "Kalman":
        det_bb_tracking = kalman_filter_tracking(copy.deepcopy(det_bb), video_n_frames, 0)
    print("Tracking finished")

    #Postprocessing
    if postprocessing == "RemoveParked":
        print("TODO: remove parked")
        print("Postprocessing finished")
    #Results
    ap = calculate_ap(det_bb_tracking, copy.deepcopy(gt_bb), 0, video_n_frames, mode='sort')
    print("Average precision: {}".format(ap))

if __name__ == '__main__':
    main()
