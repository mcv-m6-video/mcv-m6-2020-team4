import copy

from metrics.mAP import calculate_ap
from opt import parse_args_week5
from tracking.new_tracking import tracking_iou
from tracking.tracking import kalman_filter_tracking
from utils.data import read_detections_file, read_gt_txt, filter_det_confidence, number_of_images_jpg
from utils.postprocessing import clean_tracks, remove_parked
from utils.visualization import animation_tracks


def main():
    opt = parse_args_week5()
    print(opt.__dict__)

    if opt.task == 1:
        print("Starting task 1")
        task1(opt.detector, opt.trackingMethod, opt.postprocessing, opt.visualization)

    elif opt.task == 2:
        print("Starting task2")


def task1(frames_path, detections_file, gt_file, tracking_method, postprocessing=True, visualization=False):
    print("Start loading data")
    # Read gt
    gt_bb = read_gt_txt(gt_file)

    # Read and filter detections
    det_bb = read_detections_file(detections_file)
    det_bb = filter_det_confidence(det_bb, threshold=0.5)

    # Tracking
    print("Start tracking")
    video_n_frames = number_of_images_jpg(frames_path)
    if tracking_method == "MaxOverlap":
        det_bb_tracking, id_max = tracking_iou(frames_path, copy.deepcopy(det_bb), video_n_frames, mode='other')
    elif tracking_method == "Kalman":
        det_bb_tracking = kalman_filter_tracking(copy.deepcopy(det_bb), video_n_frames, 0)
        id_max = max([v[2] for v in det_bb_tracking])
    else:
        raise NotImplemented

    # Postprocessing
    if postprocessing:
        print("Starting postprocessing tracking")
        det_bb_tracking = clean_tracks(copy.deepcopy(det_bb_tracking), id_max)
        det_bb_tracking = remove_parked(copy.deepcopy(det_bb_tracking), id_max, threshold=25.0)

    # Results
    print("Start obtaining metrics")
    ap = calculate_ap(det_bb_tracking, copy.deepcopy(gt_bb), 0, video_n_frames, mode='sort')
    print("Average precision: {}".format(ap))

    # Create animation if required
    if visualization:
        print("Start storing animation")
        ini_frame = 550
        end_frame = 650
        frames_path = 'datasets/AICity_data/train/S03/c010/data'
        animation_tracks(det_bb_tracking, id_max, ini_frame, end_frame, frames_path)

    return det_bb_tracking


if __name__ == '__main__':
    main()
