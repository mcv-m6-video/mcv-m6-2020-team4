import xml.etree.ElementTree as ET
import numpy as npy
from random import shuffle


def read_nd_sort_gt(path):
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
    #Sort by frames            
    gt_bb = sorted(gt_bb, key=lambda x: x[0])
    return gt_bb


def remove_bike(gt_bb):
    gtr_bb = []
    for i in range(0, len(gt_bb)):
        if gt_bb[i][1] == 'car':
            gtr_bb.append(gt_bb[i])
    return gtr_bb



def read_detection(path):
    """
    Function that reads a .txt file for the detection
    It adds an id just in case and to make things easier
    """
    lines = open(path).read().splitlines()
    bb = []
    for l in lines:
        fields = l.split(",")    
        test_list = [int(fields[0])-1, # frame
                     'car',
                     0, #id
                     float(fields[2]), # xTopLeft
                     float(fields[3]), # yTopLeft
                     float(fields[4]) + float(fields[2]), # width + xTopLeft
                     float(fields[5]) + float(fields[3]), # height + yTopLeft
                     float(fields[6])] # confidence
        bb.append(test_list)
    return bb

