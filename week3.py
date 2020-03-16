from data import read_detections_file, read_xml_gt, filter_gt, read_xml_gt_options, filter_det_confidence
from data import save_frames, number_of_images_jpg
from faster_rcnn import inference, train
from metrics.mAP import calculate_ap
from tracking import tracking_iou, kalman_filter_tracking
from utils.utils import get_files_from_dir
from utils.visualization import animation_tracks, animation_2bb


def main():
    images_path = 'datasets/AICity_data/train/S03/c010/data'
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    gt_annot_file = 'datasets/ai_challenge_s03_c010-full_annotation.xml'
    dataset_annot_file = 'datasets/AICity_data/train/S03/c010/gt/gt.txt'
    detections_file = "datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
    save_frames(images_path)
    print("Finished saving")

    print("Task 1.1")
    # task11(images_path, gt_annot_file, config_file)
    print("Task 1.2")
    #task12(images_path, config_file, dataset_annot_file, gt_annot_file)
    print("Task 2.1")
    # task21(frames_path)
    print("Task 2.2")
    task22(gt_annot_file, detections_file, images_path)

def task11(images_path, gt_annot_file, config_file):
    files = get_files_from_dir(images_path, 'jpg')
    # we remove bikes and compute ap only for car class
    gt_bb = read_xml_gt(gt_annot_file)
    classes_to_keep = ['car']
    gt_bb = filter_gt(gt_bb, classes_to_keep)

    preds = inference(config_file, files, True)

    ap50 = calculate_ap(preds, gt_bb, 0, len(files), mode='area')
    print(ap50)

    animation_2bb('faster_on_coco', '.gif', gt_bb, preds, images_path, ini=800, )


def task12(images_path, config_file, dataset_annot_file, gt_annot_file):

    gt_bb = read_xml_gt(gt_annot_file)
    classes_to_keep = ['car']
    gt_bb = filter_gt(gt_bb, classes_to_keep)

    det_bb = train(config_file, images_path, dataset_annot_file)
    ap50 = calculate_ap(det_bb, gt_bb, 0, 2140, mode='area')
    print(ap50)

    animation_2bb('faster_finetune', '.gif', gt_bb, det_bb, images_path, ini=800, )

def task21(frames_path):
    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
    # det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_ssd512.txt")
    # det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_yolo3.txt")

    video_n_frames = number_of_images_jpg(frames_path)

    det_bb, idd = tracking_iou(det_bb, video_n_frames)
    ini_frame = 550
    end_frame = 650


    det_bb, idd = tracking_iou(det_bb, video_n_frames)
    #det_bb, idd = tracking_iou(det_bb, 5)
    ini_frame = 420
    end_frame = 480

#frame_with_2bb(det_bb, det_bb, frames_path, f_val)
    animation_tracks(det_bb, idd, ini_frame, end_frame, frames_path)


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
    original_ap = calculate_ap(det_bb, gt_bb, 0, video_n_frames, mode = ap_mode)
    # Constant velocity
    det_bb_vel = kalman_filter_tracking(det_bb, video_n_frames, 0)
    # Constant acceleration
    det_bb_acc = kalman_filter_tracking(det_bb, video_n_frames, 1)
    #Print ap results
    print("Original ap: {}".format(original_ap))
    print("Ap after tracking with constant velocity: {}".format(calculate_ap(det_bb_vel, gt_bb, 0, video_n_frames, mode = ap_mode)))
    print("Ap after tracking with constant acceleration: {}".format(calculate_ap(det_bb_acc, gt_bb, 0, video_n_frames, mode = ap_mode)))
    #Obtain animation
    max_id_vel = max(det_bb_vel, key=lambda x: x[2])[2]
    max_id_acc = max(det_bb_acc, key=lambda x: x[2])[2]
    ini_frame = 550
    end_frame = 650
    animation_tracks(det_bb_vel, max_id_vel, ini_frame, end_frame, frames_path)
    animation_tracks(det_bb_acc, max_id_acc, ini_frame, end_frame, frames_path)

if __name__ == '__main__':
    main()
