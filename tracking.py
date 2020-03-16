import numpy as np

from metrics.iou import bbox_iou
from utils.sort import Sort


def tracking_iou(det_bb, video_n_frames):
    """
    This function assigns a track value to an object by assgining the same track
    as in the frame before. To compute it, it takes the maximum overlaping bounding box from
    from the frame n+1.
    """
    idd = 0
    lst_det = [item[0] for item in det_bb]
    for f_val in range(0,video_n_frames-1):
        frame_n_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == f_val]
        frame_n1_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == f_val+1]
        # First we check if there are any overlaped detections and remove them
        for i in range(0, len(frame_n_bb)):
            length = len(frame_n_bb)
            for j in range(i+1, length):
                if length == j:
                    break
                iou = bbox_iou(frame_n_bb[i][3:7], frame_n_bb[j][3:7])
                if iou > 0.9:
                    frame_n_bb.pop(j)
                    length = length - 1
        #compare the bb from frame n to n+1
        for j in range(0, len(frame_n_bb)):
            ious = []
            for k in range(0, len(frame_n1_bb)):
                iou = bbox_iou(frame_n_bb[j][3:7], frame_n1_bb[k][3:7])
                ious.append(iou)
            arg_max = np.argmax(ious)
            #on the first frame we assign new ids to the objects in the image
            if np.max(ious) > 0.6:
                if f_val == 0:
                    frame_n_bb[j][2] = (j+1)
                    frame_n1_bb[arg_max][2] = (j+1)
                    idd = idd +1
                #for the following frames we have 2 scenarios:
                else:
                    # scenario 1: a new object appears into the scene and it has not been labeled
                    if frame_n_bb[j][2] == 0:
                        frame_n_bb[j][2] = idd
                        frame_n1_bb[arg_max][2] = idd
                        idd = idd +1
                    # scenario 2: the object was already in the scene and we need to assign to the
                    # position in the next frame the same label frame_n_bb[j][2] will already
                    # have and id different from 0 assgned
                    else:
                        frame_n1_bb[arg_max][2] = frame_n_bb[j][2]
                frame_n1_bb.pop(arg_max)
                if len(frame_n1_bb)==0:
                    break
#        print(f_val)
    return det_bb, idd

def kalman_filter_tracking(det_bb, video_n_frames, model_type):
    """
    This function assigns a track value to an object by using kalman filters.
    It also adjust bounding box coordinates based on tracking information.
    """
    bb_id_updated = []
    tracker = Sort(model_type = model_type)
    for frame_num in range(0,video_n_frames):
        #Get only bb of current frame
        dets_all_info = list(filter(lambda x: x[0] == frame_num, det_bb))
        dets = np.array([[bb[3], bb[4], bb[5], bb[6]] for bb in dets_all_info]) #[[x1,y1,x2,y2]]
        #Apply kalman filtering
        trackers = tracker.update(dets)
        #Obtain id and bb in correct format
        for bb_dets, bb_update in zip(dets_all_info, trackers):
            bb_id_updated.append([bb_dets[0], bb_dets[1], int(bb_update[4]), bb_update[0], bb_update[1], bb_update[2], bb_update[3]])
    return bb_id_updated
