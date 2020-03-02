import numpy as np
import cv2

from data import read_detections, load_annots, filter_annots, add_gauss_noise_to_bboxes, load_flow_data, process_flow_data

from utils.visualize import flow_to_hsv, visualize_flow, flow_to_color

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tlx, tly, brx, bry, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_optical_metrics(prediction, gt, thr=3):

    # discard occlusions
    occ = gt[:, :, 2] != 0

    diff = ((gt[..., :2] - prediction[..., :2]) ** 2)[occ]
    error = np.sqrt(diff[:, 0] + diff[:, 1])
    msen = error.mean()
    psen = (error > thr).sum() / error.size * 100

    return msen, psen


def compute_ap(precision, recall):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0


def evaluate(sorted_boxes, labels, ious):
    # sort predictions by confidence
    m_ap = 0.0
    preds = {}
    for frame, boxes in sorted_boxes.items():
        sorted_boxes = sorted(boxes, key=lambda x: x[-1], reverse=True)
        preds[frame] = sorted_boxes


    m_ap = np.mean([v for k, v in mAP_all_frames.items()])
    return m_ap

if __name__ == '__main__':
    dataset_folder = 'datasets/AICity_data/train/S03/c010/'
    #gt_file = dataset_folder + 'gt/gt.txt'

    mask_predictions = dataset_folder + 'det/det_mask_rcnn.txt'
    ssd_predictions = dataset_folder + 'det/det_ssd512.txt'
    yolo_predictions = dataset_folder + 'det/det_yolo3.txt'

    #gt = read_detections(gt_file)
    # df_mask = read_detections(mask_predictions)
    # df_ssd = read_detections(ssd_predictions)
    # df_yolo = read_detections(yolo_predictions)

    ious = [0.5, 0.75, 0.95]

    classes = ['car', ]
    annots = load_annots("datasets/ai_challenge_s03_c010-full_annotation.xml")
    annots = filter_annots(annots, classes=classes)

    noisy_gt = add_gauss_noise_to_bboxes(annots, 0.2)

    flows = load_flow_data("datasets/results/")
    gt = load_flow_data("datasets/results/gt/")
    flows = process_flow_data(flows)
    gt = process_flow_data(gt)

    for flow, gt in zip(flows, gt):
        msen, psen = compute_optical_metrics(flow, gt)
        print(f"MSEN: {msen}")
        print(f"PSEN: {psen}")

        visualize_flow(flow, simple=True)
        visualize_flow(gt, simple=True)

        # Better for dense optical flow
        flow_color = flow_to_color(flow[..., :2], convert_to_bgr=False)
        flow_color_gt = flow_to_color(gt[..., :2], convert_to_bgr=False)

        visualize_flow(flow_color)
        visualize_flow(flow_color_gt, suffix="_gt")
        cv2.waitKey(0)

        hsv_flow = flow_to_hsv(flow)
        hsv_flow_gt = flow_to_hsv(gt)

        visualize_flow(hsv_flow, hsv_format=True)
        visualize_flow(hsv_flow_gt, suffix="_gt", hsv_format=True)
