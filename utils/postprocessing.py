import numpy as np

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

def remove_parked(det_bb, idd, threshold = 2.0):
    """
    Remove the parked cars. Cars are detected as parked if the std of the centroid
    of its position accross frames is lower than a threshold.
    """
    # A detection still has a 0 label track then it is one of the removed overlaps
    det_bb_clean = [det_bb[i] for i, detection in enumerate(det_bb) if detection[2]!=0]

    #Detect non parked cars
    new_dets = []
    for i in range(1,idd):
        appearances = [det for det in det_bb_clean if det[2] == i]
        if len(appearances) != 0:
            xs_left = np.array([app[3] for app in appearances])
            ys_top = np.array([app[4] for app in appearances])
            xs_right = np.array([app[5] for app in appearances])
            ys_bottom = np.array([app[6] for app in appearances])
            xs_center = xs_right - xs_left
            ys_center = ys_bottom - ys_top
            std = np.mean([np.std(xs_center), np.std(ys_center)])
            if std >= threshold:
                for app in appearances:
                    new_dets.append(app)
    return new_dets
