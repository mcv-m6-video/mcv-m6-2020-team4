import numpy as np

from metrics.iou import bbox_iou


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
                if iou > 0.7:
                    frame_n_bb.pop(j)
                    length = length - 1
        #compare the bb from frame n to n+1
        for j in range(0, len(frame_n_bb)):
            if f_val == 0:
                frame_n_bb[j][2] = (j+1)
#                frame_n1_bb[arg_max][2] = (j+1)
                idd = idd +1
            else:
                ious = []
                for k in range(0, len(frame_n1_bb)):
                    iou = bbox_iou(frame_n_bb[j][3:7], frame_n1_bb[k][3:7])
                    ious.append(iou)
                arg_max = np.argmax(ious)
                #on the first frame we assign new ids to the objects in the image
                if np.max(ious) > 0.2:
                    #for the following frames we have 2 scenarios: 
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
    det_bb_clean = [det_bb[i] for i, detection in enumerate(det_bb) if detection[2]!=0]
    return det_bb_clean, idd

#def tracking_iou(det_bb, video_n_frames):
#    """
#    This function assigns a track value to an object by assgining the same track
#    as in the frame before. To compute it, it takes the maximum overlaping bounding box from
#    from the frame n+1.
#    """
#    idd = 0
#    lst_det = [item[0] for item in det_bb]
#    for f_val in range(0,video_n_frames-1):
#        frame_n_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == f_val]
#        frame_n1_bb = [det_bb[i] for i, num in enumerate(lst_det) if num == f_val+1]
#        # First we check if there are any overlaped detections and remove them
#        for i in range(0, len(frame_n_bb)):
#            length = len(frame_n_bb)
#            for j in range(i+1, length):
#                if length == j:
#                    break
#                iou = bbox_iou(frame_n_bb[i][3:7], frame_n_bb[j][3:7])
#                if iou > 0.9:
#                    frame_n_bb.pop(j)
#                    length = length - 1
#        #compare the bb from frame n to n+1
#        for j in range(0, len(frame_n_bb)):
#            ious = []
#            for k in range(0, len(frame_n1_bb)):
#                iou = bbox_iou(frame_n_bb[j][3:7], frame_n1_bb[k][3:7])
#                ious.append(iou)
#            arg_max = np.argmax(ious)
#            
#            #on the first frame we assign new ids to the objects in the image
#            if f_val == 0:
#                frame_n_bb[j][2] = (j+1)
#                frame_n1_bb[arg_max][2] = (j+1)
#                idd = idd +1
#            else:
#                #for the following frames we have 2 scenarios: 
#                if np.max(ious) > 0.5:
#                # scenario 1: the object was already in the scene and we need to assign to the
#                # position in the next frame the same label frame_n_bb[j][2] will already
#                # have and id different from 0 assgned
#                    frame_n1_bb[arg_max][2] = frame_n_bb[j][2]  
#                    frame_n1_bb.pop(arg_max)
#                # scenario 2: a new object appears into the scene and it has not been labeled
#                else:   
#                    if frame_n_bb[j][2] == 0:
#                        frame_n_bb[j][2] = idd
#                        frame_n1_bb[arg_max][2] = idd
#                        idd = idd +1   
#            
#                if len(frame_n1_bb)==0:
#                    break       
##        print(f_val) 
##    det_bb_clean = [det_bb[i] for i, detection in enumerate(det_bb) if detection[2]!=0]
#    return det_bb, idd