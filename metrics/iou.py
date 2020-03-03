from random import shuffle
import numpy as np
import copy

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


def compute_miou_list_bb(frame_det_bb, frame_gt_bb):
    miou = 0
    n_gt = len(frame_gt_bb)
    for f_det in frame_det_bb:

        ious = []

        if len(frame_gt_bb) == 0:
            break

        for f_gt in frame_gt_bb:
            iou = bbox_iou(f_det[3:], f_gt[3:])
            ious.append(iou)

        arg_max = np.argmax(ious)
        if ious[arg_max] > 0.5:
            miou += np.max(ious)
            frame_gt_bb.pop(arg_max)
    if n_gt == 0:
        return 0
    else:
        return miou/n_gt




def frame_miou(frame_det_bb, frame_gt_bb, confidence):

    if confidence:
        frame_det_bb = sorted(frame_det_bb, key=lambda x: x[-1], reverse=True)
        frame_det_bb = [item[:-1] for item in frame_det_bb]
        miou = compute_miou_list_bb(frame_det_bb, frame_gt_bb)

    else:
        miou = 0
        #Random shuffle
        for i in range(0, 10):
            shuffle(frame_det_bb)
            miou += compute_miou_list_bb(copy.deepcopy(frame_det_bb), copy.deepcopy(frame_gt_bb))
        miou = miou/10
        #Sorted by area
#       frame_det_bb = sorted(frame_det_bb, key=lambda x: (x[5]-x[3])*(x[5]-x[3]), reverse=True)
#       compute_miou_list_bb(frame_det_bb, frame_gt_bb)

    return miou
