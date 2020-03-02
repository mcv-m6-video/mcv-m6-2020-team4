import numpy as np
from random import shuffle
import data_sara


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



def calc_AP(precision, recall):
    precision = np.array(precision)
    recall = np.array(recall)
    ap = 0
    for th in np.arange(0, 1.1, 0.1):
        if np.sum(recall>=th) == 0:
            p = 0
        else:
            p = np.max(precision[recall>= th])
        ap = ap + p/11.0
    return ap

def frame_AP(n_gt, f_det_bb, frame_gt_bb):
    tp = []
    precision = []
    recall = []    
    for f_det in f_det_bb:
        ious = []
        correct = False
        
        if len(frame_gt_bb) == 0:
            continue

        for f_gt in frame_gt_bb:
            iou = bbox_iou(f_det[3:], f_gt[3:])
            ious.append(iou)
            
        arg_max = np.argmax(ious)
        if ious[arg_max]>0.5:
            frame_gt_bb.pop(arg_max)
            correct = True

        tp.append(correct)
            
        precision.append(tp.count(True)/len(tp))    
        recall.append(tp.count(True)/n_gt)
        
    ap = calc_AP(precision, recall)            
    return ap


def calculate_ap(det_bb, gt_bb, confidence):
    lst_gt = [item[0] for item in gt_bb]
    lst_det = [item[0] for item in det_bb]
    
    last_frame = np.max(lst_gt)+1
    
    i = 0
    AP = 0
    for f_val in range(0, last_frame):
        frame_gt_bb = [gt_bb[i] for i, num in enumerate(lst_gt) if num == f_val]
        n_gt = len(frame_gt_bb)
        frame_det_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == f_val]
        
        if confidence:
            frame_det_bb = sorted(frame_det_bb, key=lambda x: x[-1], reverse = True)
            f_det_bb = [item[:-1] for item in frame_det_bb]
            AP = AP + frame_AP(n_gt, f_det_bb, frame_gt_bb)
        else: 
            f_ap = 0
            for i in range(0,10):
                f_ap = f_ap + frame_AP(n_gt, shuffle(frame_det_bb), frame_gt_bb)
                
            AP = AP + f_ap/10
    AP = AP/last_frame            
    return AP
            

path = "./AICity_data/train/S03/c010/det/det_mask_rcnn.txt"

det_bb = data_sara.read_detection(path)

path = "ai_challenge_s03_c010-full_annotation.xml"

gt_bb = data_sara.read_nd_sort_gt(path)

gt_bb = data_sara.remove_bike(gt_bb)

mAP = calculate_ap(det_bb, gt_bb, confidence = True)

print('rcnn', mAP)


path = "./AICity_data/train/S03/c010/det/det_ssd512.txt"

det_bb = data_sara.read_detection(path)

mAP = calculate_ap(det_bb, gt_bb, confidence = True)

print('ssd', mAP)


path = "./AICity_data/train/S03/c010/det/det_yolo3.txt"

det_bb = data_sara.read_detection(path)

mAP = calculate_ap(det_bb, gt_bb, confidence = True)

print('yolo3', mAP)




