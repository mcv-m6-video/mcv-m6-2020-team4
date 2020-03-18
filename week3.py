import cv2
import matplotlib.pylab as plt
import numpy as np
import imageio
import copy


from data import read_detections_file, read_xml_gt, filter_gt, read_xml_gt_options, filter_det_confidence
from data import save_frames, number_of_images_jpg
#from faster_rcnn import inference, train
from metrics.mAP import calculate_ap
from tracking import tracking_iou, kalman_filter_tracking
from utils.utils import get_files_from_dir
from utils.visualization import animation_tracks, animation_2bb



def main():
    images_path = 'datasets/AICity_data/train/S03/c010/data'
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    config_file_retina = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    gt_annot_file = 'datasets/ai_challenge_s03_c010-full_annotation.xml'
    dataset_annot_file = 'datasets/AICity_data/train/S03/c010/gt/gt.txt'
    detections_file_mask = "datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
    detections_file_ssd = "datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"
    detections_file_yolo = "datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"
    detections_file_faster = "datasets/AICity_data/train/S03/c010/det/our_results_coco_faster.txt"
    detections_file_retina = "datasets/AICity_data/train/S03/c010/det/our_results_coco_retina.txt"
    detections_file_finetune_faster = "datasets/AICity_data/train/S03/c010/det/our_results_finetune_faster.txt"
    detections_file_retina = "datasets/AICity_data/train/S03/c010/det/our_results_coco_retina.txt"
    save_frames(images_path)

    print("Finished saving")
    """
    print("Task 1.1 faster")
    task11(images_path, gt_annot_file, config_file)
    print("Task 1.1 retina")
    task11(images_path, gt_annot_file, config_file_retina)

    print("Task 1.2 faster")
    task12(images_path, config_file, dataset_annot_file, gt_annot_file)
    print("Task 2.1 retina")
    task12(images_path, config_file_retina, dataset_annot_file, gt_annot_file)
    """

    # task21(gt_annot_file, detections_file, images_path)
    print("Task 2.2")
    task22(gt_annot_file, detections_file_finetune_faster, images_path)
    # print("Task 2.3")
#    task23()



def task23():
    mm.metrics.idf1()

"""

def task11(images_path, gt_annot_file, config_file):
    files = get_files_from_dir(images_path, 'jpg')
    # we remove bikes and compute ap only for car class
    gt_bb = read_xml_gt(gt_annot_file)
    classes_to_keep = ['car']
    gt_bb = filter_gt(gt_bb, classes_to_keep)

    filename = "our_results_coco_{}.txt".format("faster" if "faster" in config_file else "retina")
    preds = inference(config_file, files, save_results=True, out_filename=filename)

    ap50 = calculate_ap(preds, gt_bb, 0, len(files), mode='area')
    print(ap50)

    filename_gif ="faster_on_coco" if "faster" in config_file else "retina_on_coco"
    animation_2bb(filename_gif, '.gif', gt_bb, preds, images_path, ini=800, )


def task12(images_path, config_file, dataset_annot_file, gt_annot_file):

    gt_bb = read_xml_gt(gt_annot_file)
    classes_to_keep = ['car']
    gt_bb = filter_gt(gt_bb, classes_to_keep)

    filename = "our_results_finetune_{}.txt".format("faster" if "faster" in config_file else "retina")
    det_bb = train(config_file, images_path, dataset_annot_file, out_filename=filename)
    ap50 = calculate_ap(det_bb, gt_bb, 0, 2140, mode='area')
    print(ap50)

    filename_gif ="faster_finetune" if "faster" in config_file else "retina_finetune"
    animation_2bb(filename_gif, '.gif', gt_bb, det_bb, images_path, ini=800, )
"""
def task21(gt_annot_file, detections_file, frames_path):
    ap_mode = 'area'
    #Read and filter detections
#    det_bb = read_detections_file(detections_file)

#    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_ssd512.txt")
#    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_yolo3.txt")

    det_bb = filter_det_confidence(det_bb, threshold = 0.5)
    #Read and filter gt
    gt_bb = read_xml_gt_options(gt_annot_file, False, False)
    gt_bb = filter_gt(gt_bb, ["car"])

    video_n_frames = number_of_images_jpg(frames_path)

    original_ap = calculate_ap(det_bb, gt_bb, 0, video_n_frames, mode = ap_mode)

    det_bb_max_iou, idd = tracking_iou(copy.deepcopy(det_bb), video_n_frames)

    ap_max_iou = calculate_ap(det_bb_max_iou, gt_bb, 0, video_n_frames, mode=ap_mode)

    print("Original ap: {}".format(original_ap))
    print("Ap after tracking with maximum IoU: {}".format(ap_max_iou))
#    ini_frame = 420
#    end_frame = 580
#    animation_tracks(det_bb1, idd, ini_frame, end_frame, frames_path)


def task22(gt_annot_file, detections_file, frames_path):
    ap_mode = 'area'
    #Read and filter detections
    det_bb = read_detections_file(detections_file)
    det_bb = filter_det_confidence(det_bb, threshold = 0.5)
    #Read and filter gt
    gt_bb = read_xml_gt_options(gt_annot_file, False, False)
    gt_bb = filter_gt(gt_bb, ["car"])
    #Calculate original ap
    video_n_frames = number_of_images_jpg(frames_path)
    # Constant velocity
    det_bb_vel = kalman_filter_tracking(det_bb, video_n_frames, 0)
    # Constant acceleration
    det_bb_acc = kalman_filter_tracking(det_bb, video_n_frames, 1)
    #Print ap results
    print("Original ap: {}".format(calculate_ap(det_bb, gt_bb, 0, video_n_frames, mode = ap_mode)))
    print("Ap after tracking with constant velocity: {}".format(calculate_ap(det_bb_vel, gt_bb, 0, video_n_frames, mode = ap_mode)))
    print("Ap after tracking with constant acceleration: {}".format(calculate_ap(det_bb_acc, gt_bb, 0, video_n_frames, mode = ap_mode)))
    #Obtain animation
    max_id_vel = max(det_bb_vel, key=lambda x: x[2])[2]
    max_id_acc = max(det_bb_acc, key=lambda x: x[2])[2]
    ini_frame = 550
    end_frame = 650
    #animation_tracks(det_bb_vel, max_id_vel, ini_frame, end_frame, frames_path)
    #animation_tracks(det_bb_acc, max_id_acc, ini_frame, end_frame, frames_path)

if __name__ == '__main__':
    main()
