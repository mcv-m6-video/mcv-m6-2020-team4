import numpy as np

from metrics.iou import bbox_iou
from utils.sort import Sort


def remove_overlaps(frame_n_bb):
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
    return frame_n_bb



def update_track(det_frame, idd, det_bb, lst_det, frame):
    """
    Cheecks if a bounding box is already labeled in the past 5 frames.
    """
    n_past_frames = np.min([5,frame])
    for past_frame in range(frame-n_past_frames, frame):
        # If the past bb is labeled with 0 it is a removed bb on the remove_overlaps
        past_frame_bb = [det_bb[i] for i,num in enumerate(lst_det) if (num == past_frame and det_bb[i][2]!=0)]
        ious =[]
        for i_past_frame_bb in past_frame_bb:
            iou = bbox_iou(det_frame[3:7], i_past_frame_bb[3:7])
            ious.append(iou)
        if len(ious)==0:
            continue
        else:
            arg_max = np.argmax(ious)
            if np.max(ious)>=0.4:
                det_frame[2] = past_frame_bb[arg_max][2]
                break
            else:
                continue
    # If after checking with the past frames it still does not have a label then
    # we consider it corresponds to a new track.
    if det_frame[2] == 0:
        det_frame[2] = idd
        idd += 1
    return det_frame, idd


def clean_tracks(det_bb, idd):
    # A detection still has a 0 label track then it is one of the removed overlaps
    det_bb_clean = [det_bb[i] for i, detection in enumerate(det_bb) if detection[2]!=0]
    # If a detection lasts less than 5 frames it is not considered as a detection
    det_bb_clean = sorted(det_bb_clean, key=lambda x: x[2])
    lst_det = [item[2] for item in det_bb_clean]
    new_clean_bb =[]
    new_id = 1
    for i in range(1,idd):
        idd_n = len([ids for ids in lst_det if ids == i])
        if idd_n<=5:
            continue
        else:
            ids_to_keep = np.where(np.array(lst_det) == i)[0]
            for id_kept in ids_to_keep:
                det_bb_clean[id_kept][2] = new_id
                new_clean_bb.append(det_bb_clean[id_kept])
            new_id += 1
    new_clean_bb = sorted(new_clean_bb, key = lambda x: x[0])
    return new_clean_bb


def tracking_iou(det_bb, video_n_frames):
    idd = 0
    lst_det = [item[0] for item in det_bb]

    for frame in range(0, video_n_frames):
        # For each frame we get all the bounding boxes
        frame_n_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == frame]
        # Remove the overlaping bounding boxes on the same frame
        frame_n_bb = remove_overlaps(frame_n_bb)

        for det_frame in frame_n_bb:
            # For the first frame we label with a new label each bounding box
            if frame == 0:
                det_frame[2] = idd
                idd += 1
            else:

                det_frame, idd = update_track(det_frame, idd, det_bb, lst_det, frame)
    # A detection still has a 0 label track then it is one of the removed overlaps
    det_bb_clean = clean_tracks(det_bb, idd)
    return det_bb_clean, idd




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
            bb_id_updated.append([bb_dets[0], bb_dets[1], int(bb_update[4]), bb_update[0], bb_update[1], bb_update[2], bb_update[3], bb_dets[7]])
    return bb_id_updated
