import numpy as np

from data import read_detections, load_annots, filter_annots, add_noise_to_boxes


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

    noisy_gt = add_noise_to_boxes(annots)

    maps = evaluate(noisy_gt, annots, ious)

