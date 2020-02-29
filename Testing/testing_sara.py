import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import imageio
import copy


# Function to extract frames 
def FrameCapture(path): 
    """
    To reconstruct the video with the bounding boxes we need to extract the frames
    and save them somewhere
    """
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

        
path = "C:/Users/Sara/Datos/Master/M6/Week1/AICity_data/train/S03/c010"

"SAVE FRAMES"
#FrameCapture(path)


def read_nd_sort_gt(path):
    """
    Reads the .xml file for the GT for the cars and sorts the 
    detections by frames
    """
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
    #Sort by frames            
    gt_bb = sorted(gt_bb, key=lambda x: x[0])
    return gt_bb



path = "ai_challenge_s03_c010-full_annotation.xml"

gt_bb = read_nd_sort_gt(path)


#check that all the cars in frame 0 are detected
frame1 = cv2.imread('C:/Users/Sara/Datos/Master/M6/Week1/AICity_data/train/S03/c010/data/frame340.jpg')

for i in range(0,10):
    
    cv2.rectangle(frame1, (int(gt_bb[i][2]), int(gt_bb[i][3])), 
                          (int(gt_bb[i][4]), int(gt_bb[i][5])), (255,0,0), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame1,"1",(int(gt_bb[i][2]), int(gt_bb[i][3])-10), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    
frame1 = cv2.resize(frame1, (int(1920/2),int(1080/2)))
cv2.imshow("image", frame1)
cv2.waitKey(0)          



#%%


def animation_bb(name, form, bb_cords, frame_path, fps, seconds, width, height):
    """
    Records a video for a single bounding box.
    """
    if len(bb_cords) == 7:
        confid = True
    else:
        confid = False
                
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./' + name + form, fourcc, float(FPS), (width, height))
    ini = 390
    lst2 = [item[0] for item in bb_cords]
    images = []
    
    for i in range(FPS*seconds):
        f_val = i + ini
        frame1 = cv2.imread((frame_path + 'frame{}.jpg').format(f_val))
        
        args = [i for i, num in enumerate(lst2) if num == f_val]
        for ar in args:        
            cv2.rectangle(frame1, (int(bb_cords[ar][2]), int(bb_cords[ar][3])), 
                                  (int(bb_cords[ar][4]), int(bb_cords[ar][5])), (0,255,0), 2)
            if confid:
                cv2.putText(frame1,str(bb_cords[ar][6]),
                            (int(bb_cords[i][2]), int(bb_cords[i][3])-10), 
                            font, 0.5, (255,0,0), 2, cv2.LINE_AA)
    
        frame1 = cv2.resize(frame1, (width,height))
        
        if form == '.gif':
            images.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        else:
            video.write(frame1)
    
    if form == 'gif':
        imageio.mimsave('./' + name + form, images)        
    else:    
        video.release()
    return frame1
    
    
    
#width = int(1920/4)
#height = int(1080/4)
#FPS = 10
#seconds = 20
#name = 'holahola'
#form = '.avi'
#frame_path = "C:/Users/Sara/Datos/Master/M6/Week1/AICity_data/train/S03/c010/data/"
#
#
#
#a = animation_bb(name, form, gt_bb, frame_path, FPS, seconds, width, height)


#%%
    
def gauss_noisy_bb(gt_bb):
    """
    Function that adds noise to the image 
    WE WILL HAVE TO CHECK THE EFFECT OF CHANGING THE NOISE LEVEL
    """
    np.random.seed(2373)
    noisy_bb = copy.deepcopy(gt_bb)
                        
    for i in range(0,len(gt_bb)):
        for j in range(0, 4):
            noisy_bb[i][2+j] = gt_bb[i][2+j] + float(np.random.normal(0,5,1))
    return noisy_bb
    


def animation_2bb(name, form, gt_bb, bb_cords, frame_path, fps, seconds, width, height):
    """
    This function records a video of some frames with both the GT (green) 
    and detection (blue) bounding boxes. If we have a confidence value that number 
    is added on top of the bounding box.
    Input
        Name: Name of the file to save
        form: format of the file, it can be .avi or .gif (. must be included)
        gt_bb: ground truth bounding boxes in the same format as reed
        bb_cords: bounding box for the detection        
    """
    
    #in case we have a confidence value in the detection
    if len(bb_cords) == 7:
        confid = True
    else:
        confid = False
                
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./' + name + form, fourcc, float(FPS), (width, height))
    ini = 390
    lst_gt = [item[0] for item in gt_bb]
    lst_nogt = [item[0] for item in bb_cords]
    images = []
    
    for i in range(FPS*seconds):
        f_val = i + ini
        frame1 = cv2.imread((frame_path + 'frame{}.jpg').format(f_val))
        
        args_gt = [i for i, num in enumerate(lst_gt) if num == f_val]
        for ar in args_gt:        
            #Ground truth bounding box in green
            cv2.rectangle(frame1, (int(gt_bb[ar][2]), int(gt_bb[ar][3])), 
                                  (int(gt_bb[ar][4]), int(gt_bb[ar][5])), (0,255,0), 2)
            
            
        args_nogt = [i for i, num in enumerate(lst_nogt) if num == f_val]
        for ar in args_nogt:        
            #guessed GT in blue
            cv2.rectangle(frame1, (int(bb_cords[ar][2]), int(bb_cords[ar][3])), 
                                  (int(bb_cords[ar][4]), int(bb_cords[ar][5])), (255,0,0), 2)
            
            if confid:
                cv2.putText(frame1,str(bb_cords[ar][6]) + " %",
                            (int(bb_cords[i][2]), int(bb_cords[i][3])-10), 
                            font, 0.5, (255,0,0), 2, cv2.LINE_AA)
    
        frame1 = cv2.resize(frame1, (width,height))
        
        if form == '.gif':
            images.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        else:
            video.write(frame1)
    
    if form == 'gif':
        imageio.mimsave('./' + name + form, images)        
    else:    
        video.release()
    return frame1  


width = int(1920/4)
height = int(1080/4)
FPS = 10
seconds = 20
name = 'holahola'
form = '.avi'
frame_path = "./AICity_data/train/S03/c010/data/"



#a = animation_2bb(name, form, gt_bb, gauss_noisy_bb(gt_bb), frame_path, FPS, seconds, width, height)


path = "./AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
def read_detection(path):
    """
    Function that reads a .txt file for the detection
    It adds an id just in case and to make things easier
    """
    lines = open(path).read().splitlines()
    bb = []
    for l in lines:
        fields = l.split(",")    
        test_list = [float(fields[0]), # frame
                     0, #id
                     float(fields[2]), # xTopLeft
                     float(fields[3]), # yTopLeft
                     float(fields[4]), # xBottomRight
                     float(fields[5]), # yBottomRight
                     float(fields[6])] # confidence
        bb.append(test_list)
    return bb

rcnn_bb = read_detection(path)





"""
IN PROGRESS
"""
iou_th = 0.5

nogt_bb = gauss_noisy_bb(gt_bb)

lst_gt = [item[0] for item in gt_bb]
lst_nogt = [item[0] for item in nogt_bb]

last_frame = np.max(lst_gt)


all_ious = []
for f_val in range(0, last_frame):
    args_gt = [i for i, num in enumerate(lst_gt) if num == f_val]
    args_nogt = [i for i, num in enumerate(lst_nogt) if num == f_val]
    for i_nogt in args_nogt:
        correct = False
        ious = []
        for i_gt in args_gt:
            iou = bbox_iou(gt_bb[i_gt][2:6], nogt_bb[i_nogt][2:6])
            ious.append(iou)
            if iou>iou_th:
                correct = True
        if correct == False:
            print('hola')
        all_ious.append(max(ious))
    
        


