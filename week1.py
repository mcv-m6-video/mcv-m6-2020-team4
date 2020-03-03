from data import load_annots, filter_annots
from data import read_detections_file, read_xml_gt, filter_gt, generate_noisy_annotations, load_flow_data, \
    process_flow_data
from metrics.iou import frame_miou
from metrics.mAP import calculate_ap
from metrics.optical_flow import compute_optical_metrics
from utils.optical_flow_visualization import visualize_flow, flow_to_color, flow_to_hsv
from utils.visualization import animate_iou, animation_2bb, plot_animation
import numpy as np




def compute_map(gt, detections_file):
    det_bb = read_detections_file(detections_file)
    return calculate_ap(det_bb, gt, confidence=True)


def main():
    print("Task 1")
    task1("datasets/ai_challenge_s03_c010-full_annotation.xml")
    print("Task 2")
    task2("datasets/ai_challenge_s03_c010-full_annotation.xml", 390)
    print("Task 3")
    flow1, gt1 = task3("datasets/results/LKflow_000045_10.png", "datasets/results/gt/000045_10.png")
    flow2, gt2 = task3("datasets/results/LKflow_000157_10.png", "datasets/results/gt/000157_10.png")

    print("Task 4")
    task4(flow1, gt1)
    task4(flow2, gt2)


def task1(gt_file):
    gt = read_xml_gt(gt_file)
    classes_to_keep = ['car']
    gt = filter_gt(gt, classes_to_keep)

    det_bb = generate_noisy_annotations(gt)
    
    lst_gt = [item[0] for item in gt]
    last_frame = np.max(lst_gt)
    
    miou = 0
    for f_val in range(0, last_frame):
        frame_gt_bb = [gt[i] for i, num in enumerate(gt) if num[0] == f_val]
        
        frame_det_bb = [det_bb[i] for i, num in enumerate(det_bb) if num[0] == f_val]
        miou += frame_miou(frame_det_bb, frame_gt_bb, confidence=False)
        
    miou = miou/last_frame
    
#    print("noisy gt ap random: {}".format(calculate_ap(det_bb, gt, 1)))
#    print("noisy gt ap area: {}".format(calculate_ap(det_bb, gt, 2)))
    
    print("mIoU, ", miou)
    

#    preds_mask = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
#    print("maskrcnn ap: {}".format(calculate_ap(preds_mask, gt, True)))
#
#    preds_ssd = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_ssd512.txt")
#    print("ssd ap: {}".format(calculate_ap(preds_ssd, gt, True)))
#
#    preds_yolo = read_detections_file("datasets/AICity_data/train/S03/c010/det/det_yolo3.txt")
#    print("yolo ap: {}".format(calculate_ap(preds_yolo, gt, True)))




def task2(gt_file, ini_frame):
    gt_bb = read_xml_gt(gt_file)
    classes_to_keep = ['car']
    gt_bb = filter_gt(gt_bb, classes_to_keep)
    
    det_bb = generate_noisy_annotations(gt_bb)
    
    fps = 10
    seconds = 30
    animation_2bb("file", ".avi", gt_bb, det_bb, "datasets/AICity_data/train/S03/c010/data/", fps, seconds,
              ini_frame, int(1920 / 4), int(1080 / 4))
    
    mious = []
    for f_val in range(ini_frame, ini_frame+int(seconds*fps)):
        frame_gt_bb = [gt_bb[i] for i, num in enumerate(gt_bb) if num[0] == f_val]
        
        frame_det_bb = [det_bb[i] for i, num in enumerate(det_bb) if num[0] == f_val]
        mious.append(frame_miou(frame_det_bb, frame_gt_bb, confidence=False))
        
    
    frames = list(range(0, len(mious)))
    ani = plot_animation(frames, mious, 'Frame', 'mIoU', [0,1], fps)
    ani.save('test.gif')
    


def task3(image_file, gt_file):
    flow = load_flow_data(image_file)
    gt = load_flow_data(gt_file)

    flow = process_flow_data(flow)
    gt = process_flow_data(gt)

    msen, psen = compute_optical_metrics(flow, gt)
    print(f"MSEN: {msen}")
    print(f"PSEN: {psen}")

    return flow, gt


def task4(flow, gt):
    visualize_flow(flow, simple=True)
    visualize_flow(gt, suffix="_gt", simple=True)

    # Better for dense optical flow
    flow_color = flow_to_color(flow[..., :2], convert_to_bgr=False)
    flow_color_gt = flow_to_color(gt[..., :2], convert_to_bgr=False)

    visualize_flow(flow_color)
    visualize_flow(flow_color_gt, suffix="_gt")

    hsv_flow = flow_to_hsv(flow)
    hsv_flow_gt = flow_to_hsv(gt)

    visualize_flow(hsv_flow, hsv_format=True)
    visualize_flow(hsv_flow_gt, suffix="_gt", hsv_format=True)


if __name__ == '__main__':
    main()
