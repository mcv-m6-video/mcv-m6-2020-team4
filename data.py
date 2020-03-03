import copy
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import xmltodict

import random


def load_flow_data(image_file):
    return cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(np.double)


def process_flow_data(flow):
    # Taken from KITTI toolbox (flow_read.m)
    u_vec = (flow[:, :, 2] - 2 ** 15) / 64
    v_vec = (flow[:, :, 1] - 2 ** 15) / 64

    valid = flow[:, :, 0] == 1
    u_vec[~valid] = 0
    v_vec[~valid] = 0

    return np.dstack((u_vec, v_vec, valid))


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
            x1 = float(box["@xtl"])
            y1 = float(box["@ytl"])
            x2 = float(box["@xbr"])
            y2 = float(box["@ybr"])
            occluded = bool(box["@occluded"])

            annot_list.append([image_id, image_label, frame, x1, y1, x2, y2, occluded])

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


def generate_noisy_annotations(gt_bb):
    """
    Function that adds noise to the image
    WE WILL HAVE TO CHECK THE EFFECT OF CHANGING THE NOISE LEVEL
    """
    xtl_mean = np.mean([item[-4] for item in gt_bb])
    ytl_mean = np.mean([item[-3] for item in gt_bb])
    xbr_mean = np.mean([item[-2] for item in gt_bb])
    ybr_mean = np.mean([item[-1] for item in gt_bb])

    np.random.seed(2373)
    noisy_bb = copy.deepcopy(gt_bb)
    lst_gt = [item[0] for item in gt_bb]
    last_frame = np.max(lst_gt)+1

    # Remove the 5% of the bounding boxes
    args_to_keep = random.sample(range(0, len(noisy_bb)), int(len(noisy_bb) * 0.99))
    keep_bb = []
    for i in args_to_keep:
        keep_bb.append(noisy_bb[i])
    '''
    keep_bb = []
    keep_bb = copy.deepcopy(noisy_bb)
    # Change the 5% of the bounding boxes in the GT

    args_to_generate = int(len(gt_bb) * 0.2)

    lst_gt = [item[0] for item in gt_bb]
    last_frame = np.max(lst_gt)

    for i in range(0,args_to_generate):
        frame_to_insert = np.random.randint(0, last_frame)
        new_bb = gen_random_bb(xtl_mean, ytl_mean, xbr_mean, ybr_mean, 100)

        keep_bb.append([frame_to_insert, 'car', 0, new_bb[0], new_bb[1], new_bb[2], new_bb[3]])

    for i in range(0, len(keep_bb)):
        for j in range(0, 4):
            keep_bb[i][3 + j] = keep_bb[i][3 + j] + float(np.random.normal(0, 30, 1))

    keep_bb = sorted(keep_bb, key=lambda x: x[0], reverse=False)
    '''
    return keep_bb


def add_noise_to_bbox(boxes, low=-5, high=5):
    return [box + np.random.uniform(low, high, 4) for box in boxes]


def gen_random_bb(xtl_mean, ytl_mean, xbr_mean, ybr_mean, std):
    sx = 1920
    sy = 1080

    w = abs(xtl_mean - xbr_mean) + float(np.random.normal(0, std, 1))
    h = abs(ytl_mean - ybr_mean) + float(np.random.normal(0, std, 1))

    xc = np.random.uniform(h, sx - h, 1)
    yc = np.random.uniform(w, sy - w, 1)

    xtl = xc - int(w / 2)
    xbr = xc + int(w / 2)
    ytl = yc - int(h / 2)
    ybr = yc + int(h / 2)
    return [float(xtl), float(ytl), float(xbr), float(ybr)]


def read_xml_gt(path):
    """
    Reads the .xml file for the GT for the cars and sorts the
    detections by frames
    """
    tree = ET.parse(path)

    root = tree.getroot()

    gt_bb = []
    for child in root[2:]:
        for c in child:
            lista = [int(c.attrib['frame']),
                     child.attrib['label'],
                     int(child.attrib['id']),
                     float(c.attrib['xtl']),
                     float(c.attrib['ytl']),
                     float(c.attrib['xbr']),
                     float(c.attrib['ybr'])]
            gt_bb.append(lista)
    # Sort by frames
    gt_bb = sorted(gt_bb, key=lambda x: x[0])
    return gt_bb


def filter_gt(gt_bb, classes_to_keep):
    gtr_bb = []
    for i in range(0, len(gt_bb)):
        if gt_bb[i][1] in classes_to_keep:
            gtr_bb.append(gt_bb[i])
    return gtr_bb


def read_detections_file(path):
    """
    Function that reads a .txt file for the detection
    It adds an id just in case and to make things easier
    """
    lines = open(path).read().splitlines()
    bb = []
    for l in lines:
        fields = l.split(",")
        test_list = [int(fields[0]) - 1,  # frame
                     'car',
                     0,  # id
                     float(fields[2]),  # xTopLeft
                     float(fields[3]),  # yTopLeft
                     float(fields[4]) + float(fields[2]),  # width + xTopLeft
                     float(fields[5]) + float(fields[3]),  # height + yTopLeft
                     float(fields[6])]  # confidence
        bb.append(test_list)
    return bb
