import numpy as np
import cv2
import copy


from metrics.iou import bbox_iou
from utils.sort import Sort


def remove_overlaps(frame_n_bb):
    """
    This function detects if we have any overlapped detections, two bboxes for 
    the same object, and removes one.
    """
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
    Cheecks if a bounding box is already labeled in the past 3 frames
    or in the first 5 frames.
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
            
    # We also check if the bb is in the first 5 frames as they are bg and they should
    # appear in all the frames
    mm = np.min([5,frame])
    for bg_frame in range(0,mm):
        bg_frame_bb = [det_bb[i] for i,num in enumerate(lst_det) if (num == bg_frame and det_bb[i][2]!=0)]                
        ious = []
        for i_bg_frame_bb in bg_frame_bb:
            iou = bbox_iou(det_frame[3:7], i_bg_frame_bb[3:7])
            ious.append(iou)
        if len(ious)==0:
            continue
        else:
            arg_max = np.argmax(ious)
            if np.max(ious)>=0.4:
                det_frame[2] = bg_frame_bb[arg_max][2]
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
    """
    When the whole detection ends if an id is tracked with the value 0, then
    it is a removed track and we need to erase it.
    """
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
    
    

def update_track_of(flow, det_frame, det_bb, idd, lst_det, frame):
    """
    Moves a bb from the actual position to the position it occuied in the 
    past frame. If there is a bb with a large IoU we assign the same track value,
    it not we create a new track value.
    """
    
    xtl_1, ytl_1, xbr_1, ybr_1 = np.array(det_frame[3:7]).astype(int)
       
    bb_flow = flow[ytl_1:ybr_1, xtl_1:xbr_1, :]
    
    xtl = xtl_1 + np.mean(bb_flow[...,0])
    xbr = xbr_1 + np.mean(bb_flow[...,0])
    ytl = ytl_1 - np.mean(bb_flow[...,1])
    ybr = ybr_1 - np.mean(bb_flow[...,1])
    
    old_det = [xtl, ytl, xbr, ybr]

    
    # If the past bb is labeled with 0 it is a removed bb on the remove_overlaps
    past_frame_bb = [det_bb[i] for i,num in enumerate(lst_det) if (num == frame-1 and det_bb[i][2]!=0)]                
    ious =[]
    for i_past_frame_bb in past_frame_bb:
        iou = bbox_iou(old_det, i_past_frame_bb[3:7])
        ious.append(iou)
    if len(ious)==0:
        pass
    else:
        arg_max = np.argmax(ious)
        if np.max(ious)>=0.7:
            det_frame[2] = past_frame_bb[arg_max][2]
        else:
            pass
    
    if det_frame[2] == 0:
        det_frame[2] = idd
        idd += 1
    
    return det_frame, idd
    
    
def tracking_iou(frames_path, det_bb, video_n_frames, mode):
    """
    Function to assign a unique track value to each detection using maximum overlap.
    Two modes:
        - None: Just using the detections
        - of: Computing a predicted bbox using optical flow
    Returns the last idd assigned and the list of detections with the track id
    on the 3rd position.
    """
    
    idd = 0
    lst_det = [item[0] for item in det_bb]    
    
    for frame in range(0, video_n_frames):
        # For each frame we get all the bounding boxes
        frame_n_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == frame]
        # Remove the overlaping bounding boxes on the same frame
        frame_n_bb = remove_overlaps(frame_n_bb)
        
        if frame > 0 and mode == 'of':
            #past frame
            first_frame = cv2.imread((frames_path + '/frame_{:04d}.jpg').format(frame))
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            #actual frame
            second_frame = cv2.imread((frames_path + '/frame_{:04d}.jpg').format(frame + 1))
            second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(second_frame, first_frame, 
                                                None, 0.5, 3, 15, 3, 5, 1.2, 10)
                
        for det_frame in frame_n_bb:
            # For the first frame we label with a new label each bounding box
            if frame == 0:
                det_frame[2] = idd
                idd += 1
            else:
                if mode == 'of':
                    det_frame, idd = update_track_of(flow, det_frame, det_bb, idd, lst_det, frame)                                               
#                print(frame, idd)                           
                else:
                    det_frame, idd = update_track(det_frame, idd, det_bb, lst_det, frame)
                
                
    # A detection still has a 0 label track then it is one of the removed overlaps
    if mode != 'of':
        det_bb_clean = clean_tracks(det_bb, idd)        
        return det_bb_clean, idd
    else:
        return det_bb, idd


def restore_tracks(frames_path, det_bb_max_iou):
    """
    After tracking we try to restory any missed detections we might have. 
    We compute the future position of the bb using optical flow, if it doesn't 
    exist in the next frame but does exist in the following frame then add the 
    predicted bounding box.
    """
    
    new_tracks = copy.deepcopy(det_bb_max_iou)
    for its in range(0,3):
        new_tracks = copy.deepcopy(new_tracks)

        sorted_id_det = sorted(copy.deepcopy(new_tracks), key=lambda x: x[2])        
        last_id = sorted_id_det[-1][2] + 1
        
        new_tracks = []
        for idd in range(1, last_id):
            nframes = [num[0] for i, num in enumerate(sorted_id_det) if num[2] == idd]
            ndet = [num for i, num in enumerate(sorted_id_det) if num[2] == idd]
            
            for j, frame in enumerate(nframes):
                if j+1 == len(nframes):
                    new_tracks.append(ndet[j])
                else:            
                    if nframes[j+1] > nframes[j]+1:
                        #I have the bb info of this frame
                        first_frame = cv2.imread((frames_path + '/frame_{:04d}.jpg').format(frame+1))
                        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                        
                        #I don't have the bb info of this frame
                        second_frame = cv2.imread((frames_path + '/frame_{:04d}.jpg').format(frame+2))
                        second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
                        
                        flow = cv2.calcOpticalFlowFarneback(first_frame, second_frame, 
                                                            None, 0.5, 3, 15, 3, 5, 1.2, 10)
                        
                        xtl_1, ytl_1, xbr_1, ybr_1 = np.array(ndet[j][3:7]).astype(int)
                   
                        bb_flow = flow[ytl_1:ybr_1, xtl_1:xbr_1, :]
                        
                        xtl = xtl_1 + np.mean(bb_flow[...,0])
                        xbr = xbr_1 + np.mean(bb_flow[...,0])
                        ytl = ytl_1 - np.mean(bb_flow[...,1])
                        ybr = ybr_1 - np.mean(bb_flow[...,1])
                        
                        new_det = [frame + 1, 'car', idd, xtl, ytl, xbr, ybr, 0]
        #                print(ndet[j][3:7], new_det[3:7])
                        new_tracks.append(ndet[j])
                        
                        new_tracks.append(new_det)
                        
        #            elif nframes[j+1] == nframes[j]:
        #                pass
                    else:
                        new_tracks.append(ndet[j])            
            
        new_tracks = sorted(new_tracks, key = lambda x: x[0])         
        print(its)
    return new_tracks

