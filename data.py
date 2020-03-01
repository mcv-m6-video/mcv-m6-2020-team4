import os
import numpy as np
import copy

import pandas as pd
import xmltodict


def read_detections(file):
    df = pd.read_csv(file, header=None)
    df.columns = ["image_id", "patata", "x1", "y1", "x2", "y2", "confidence", "patata2", "patata3", "patata4"]
    df['box'] = df.iloc[:, 2:6].values.tolist()
    df.drop(columns=["x1", "y1", "x2", "y2", 'patata', 'patata2', 'patata3', 'patata4'], inplace=True)

    return df


def load_annots(file):
    assert os.path.exists(file)
    f = open(file, 'r')
    annots = xmltodict.parse(f.read())

    # get only the image annotations
    annots = annots['annotations']['track']

    annot_list = []
    for image in annots:
        image_id = image['@id']
        image_label = image['@label']

        for box in image['box']:
            frame = box['@frame']
            x1 = box["@xtl"]
            y1 = box["@ytl"]
            x2 = box["@xbr"]
            y2 = box["@ybr"]

            annot_list.append([image_id, image_label, frame, x1, y1, x2, y2])

    return annot_list


def filter_annots(annots, classes):
    filtered_annots = [annot for annot in annots if annot[1] in classes]

    annots = {}
    for annot in filtered_annots:
        frame = annot[0]
        if frame not in annots.keys():
            annots[frame] = []
        annots[frame].append(annot[3:])

    return annots


def add_noise_to_boxes(annots):
    noise_function = lambda x: x


def add_gauss_noise_to_bboxes(gt_bb, std):
    """
    Function that adds noise to the image
    WE WILL HAVE TO CHECK THE EFFECT OF CHANGING THE NOISE LEVEL
    """
    np.random.seed(2373)
    noisy_bb = copy.deepcopy(gt_bb)

    for i, bb in enumerate(gt_bb):
        for j, val in enumerate(bb):
            noisy_bb[i][j] = float(val) + float(np.random.normal(0,std,1))
    return noisy_bb
