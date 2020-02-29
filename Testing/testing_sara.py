import xml.etree.ElementTree as ET
import cv2 
import numpy as np

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path + "/vdo.avi") 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite(path + "/data/frame%d.jpg" % count, image) 
  
        count += 1
        print(count)


def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou

        
path = "C:/Users/Sara/Datos/Master/M6/Week1/AICity_data/train/S03/c010"

"SAVE FRAMES"
#FrameCapture(path)


def read_nd_sort_gt(path):
    

    tree = ET.parse(path)

    root = tree.getroot()
    
    
    gt_bb = []
    for child in root[2:]:
        if child.attrib['label'] == 'car':
            for c in child:
                lista = [int(c.attrib['frame']),
                         int(child.attrib['id']),
                         float(c.attrib['xtl']),
                         float(c.attrib['ytl']),
                         float(c.attrib['xbr']),
                         float(c.attrib['ybr'])]
                gt_bb.append(lista)
    gt_bb = sorted(gt_bb, key=lambda x: x[0])
    return gt_bb



path = "ai_challenge_s03_c010-full_annotation.xml"

gt_bb = read_nd_sort_gt(path)


#check that all the cars in frame 0 are detected
frame1 = cv2.imread('C:/Users/Sara/Datos/Master/M6/Week1/AICity_data/train/S03/c010/data/frame0.jpg')

for i in range(0,8):
    
    cv2.rectangle(frame1, (int(gt_bb[i][2]), int(gt_bb[i][3])), 
                          (int(gt_bb[i][4]), int(gt_bb[i][5])), (255,0,0), 2)

    
frame1 = cv2.resize(frame1, (int(1920/2),int(1080/2)))
cv2.imshow("image", frame1)
cv2.waitKey(0)            