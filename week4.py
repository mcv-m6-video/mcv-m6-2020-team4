import os
from tqdm import tqdm

import cv2
import imageio
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, mode

from data import save_frames, load_flow_data, process_flow_data
from metrics.optical_flow import compute_optical_metrics
from optical_flow import OpticalFlowBlockMatching
from utils.optical_flow_visualization import visualize_flow, flow_to_hsv, visualize_flow_v2
from utils.utils import get_files_from_dir
from utils.visualization import visualize_3d_plot
from new_tracking import tracking_iou, restore_tracks
from data import read_detections_file, filter_gt, read_xml_gt_options
from data import number_of_images_jpg
from metrics.mAP import calculate_ap
from utils.visualization import animation_tracks


from utils.utils import getVideoArray, getTrans, reconVideo, fix_border, smooth
from pyflow import pyflow

def main():
    images_path = 'datasets/AICity_data/train/S03/c010/data'

    save_frames(images_path)
    print("Finished saving")

    print("Task 1.1")
    s1_flow_gt = "datasets/results/gt/000045_10.png"
    s1_im1_path = "datasets/results/images/000045_10.png"
    s1_im2_path = "datasets/results/images/000045_11.png"
    s2_flow_gt = "datasets/results/gt/000157_10.png"
    s2_im1_path = "datasets/results/images/000157_10.png"
    s2_im2_path = "datasets/results/images/000157_11.png"
    task11(images_path, s1_flow_gt, s1_im1_path, s1_im2_path, s2_flow_gt, s2_im1_path, s2_im2_path)

    print("Task 1.2")
    task12(im1_path, im2_path, flow_gt, algorithm="pyflow")
    task12(im1_path, im2_path, flow_gt, algorithm="fb")
    print("Task 2.1")
    # task21(images_path)

    print("Task 2.2")
    #task_22("datasets/video/zoo.mp4", method="point")
    #task_22("datasets/video/zoo.mp4", method="fast")

	print("Task 3.1")
    task31()

def task11(frames_path, flow_gt, im1_path, im2_path, s2_flow_gt, s2_im1_path, s2_im2_path):
    compute_optical_flow_metrics = True
    grid_search_block_area = False
    compare_step = False

    # Load ground truth
    gt = load_flow_data(flow_gt)
    gt = process_flow_data(gt)

    gt2 = load_flow_data(s2_flow_gt)
    gt2 = process_flow_data(gt2)

    # Load frames
    first_frame = cv2.imread(im1_path)
    second_frame = cv2.imread(im2_path)

    first_frame2 = cv2.imread(s2_im1_path)
    second_frame2 = cv2.imread(s2_im2_path)

    if compute_optical_flow_metrics:
        # Compute optical flow
        flow_func = OpticalFlowBlockMatching(type="BW", block_size=21, area_search=20, error_function="SSD",
                                             window_stride=1)
        flow = flow_func.compute_optical_flow(first_frame, second_frame)
        flow2 = flow_func.compute_optical_flow(first_frame2, second_frame2)

        # Compute metrics
        msen, psen = compute_optical_metrics(flow, gt, plot_error=False)
        msen2, psen2 = compute_optical_metrics(flow2, gt2, plot_error=False)
        print("Average MSEN: {}".format(msen+msen2/2.0))
        print("Average PSEN: {}".format(psen+psen2/2.0))

        #Visualize the computed optical flow
        visualize_flow_v2(first_frame, flow)
        hsv_flow = flow_to_hsv(flow)
        visualize_flow(hsv_flow, hsv_format=True)

    if grid_search_block_area:
        area_sizes = np.array([20, 40, 60, 80, 100, 120])
        block_sizes = np.array([3, 5, 7, 9, 11, 15, 17, 19, 21])
        msens = np.zeros((len(block_sizes), len(area_sizes)))
        psens = np.zeros((len(block_sizes), len(area_sizes)))
        X, Y = np.meshgrid(area_sizes, block_sizes)
        for i, area_size in enumerate(area_sizes):
            for j, block_size in enumerate(block_sizes):
                # Compute optical flow
                flow_func = OpticalFlowBlockMatching(type="FW", block_size=block_size, area_search=area_size,
                                                     error_function="SSD", window_stride=block_size)
                flow = flow_func.compute_optical_flow(first_frame, second_frame)
                msen, psen = compute_optical_metrics(flow, gt, plot_error=False)
                print("Area size: {:2f} | Block size: {:2f} | msen: {:2f} | psen: {:2f} |".format(area_size, block_size,
                                                                                  msen, psen))
                msens[j, i] = msen
                psens[j, i] = psen
        # Plot msen grid search
        visualize_3d_plot(X, Y, msens, 'Area size', 'Block size', 'MSEN')
        # Plot psen grid search
        visualize_3d_plot(X, Y, psens, 'Area size', 'Block size', 'PEPN')

    if compare_step:
        step_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        msens = np.zeros((len(step_sizes), 1))
        psens = np.zeros((len(step_sizes), 1))

        for i, step_size in enumerate(step_sizes):
            flow_func = OpticalFlowBlockMatching(type="FW", block_size=21, area_search=20,
                                                 error_function="SSD", window_stride=step_size)
            flow = flow_func.compute_optical_flow(first_frame, second_frame)
            msen, psen = compute_optical_metrics(flow, gt, plot_error=False)
            print("Step size: {:2f} | msen: {:2f} | psen: {:2f} |".format(step_size, msen, psen))
            msens[i] = msen
            psens[i] = psen

        fig = plt.figure()
        plt.plot(step_sizes, msens,  marker = 'o')
        plt.xlabel("Step size")
        plt.ylabel("MSEN")
        plt.show()

        fig2 = plt.figure()
        plt.plot(step_sizes, psens,  marker = 'o')
        plt.xlabel("Step size")
        plt.ylabel("PEPN")
        plt.show()

def task12(im1, im2, flow_gt, algorithm='pyflow'):

    img_prev = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread(im2, cv2.IMREAD_GRAYSCALE)
    gt = load_flow_data(flow_gt)
    gt = process_flow_data(gt)

    if algorithm == 'pyflow':
        im1 = np.atleast_3d(img_prev.astype(float) / 255.)
        im2 = np.atleast_3d(img_next.astype(float) / 255.)


        # taken from https://github.com/pathak22/pyflow/blob/8f8ab9e90845f1b819b3833f995f481cb5d92166/demo.py#L25
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations,
                                             nInnerFPIterations, nSORIterations, colType)
        flow = np.dstack((u, v))


    elif algorithm == 'fb':
        flow = cv2.calcOpticalFlowFarneback(img_prev, img_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)


    else:
        raise NotImplementedError("Choose algorithm from 'pyflow' or 'fb'")

    # hsv_flow = flow_to_hsv(flow)
    # print("visualize hsv")
    # visualize_flow(hsv_flow, hsv_format=True)
    msen, psen = compute_optical_metrics(flow, gt, plot_error=False)
    print("Method: {algorithm}, msen: {msen}, psen: {psen}")








def task_21(frames_path, optical_flow_func='SSD', stab_mode='trimmed_mean', trimmed_mean_percentage=0.1):
    assert os.path.exists(frames_path)
    files = get_files_from_dir(frames_path, "jpg")
    assert len(files) != 0, "no frames in folder."

    opt_flow = OpticalFlowBlockMatching(type="FW", block_size=21, area_search=20, error_function=optical_flow_func,
                                        window_stride=1)

    accumulated_flow = np.zeros(2)
    stab = []
    unstab = []
    for idx, frame in enumerate(sorted(files)):
        if idx == 0:
            im_frame = cv2.imread(frame)
            h, w = im_frame.shape[:2]
            h, w = int(h / 4), int(w / 4)
            im_frame = cv2.resize(im_frame, (w, h))
            stab.append(cv2.cvtColor(im_frame, cv2.COLOR_BGR2RGB))
            unstab.append(cv2.cvtColor(im_frame, cv2.COLOR_BGR2RGB))
        else:
            im_frame = cv2.imread(frame)
            h, w = im_frame.shape[:2]
            h, w = int(h / 4), int(w / 4)
            im_frame = cv2.resize(im_frame, (w, h))
            unstab.append(cv2.cvtColor(im_frame, cv2.COLOR_BGR2RGB))

            motion_matrix = opt_flow.compute_optical_flow(reference_frame, im_frame)
            # motion_matrix = cv2.calcOpticalFlowFarneback(cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY),
            #                                              cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY),
            #                                              None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            if stab_mode == 'mean':
                u = motion_matrix[:, :, 0].mean()
                v = motion_matrix[:, :, 1].mean()
            elif stab_mode == 'trimmed_mean':
                u = trim_mean(motion_matrix[:, :, 0], trimmed_mean_percentage, axis=None)
                v = trim_mean(motion_matrix[:, :, 1], trimmed_mean_percentage, axis=None)
            elif stab_mode == 'median':
                u, v = np.median(motion_matrix[:, :, 0]), np.median(motion_matrix[:, :, 1])
            elif stab_mode == 'mode':
                us, vs = cv2.cartToPolar(motion_matrix[:, :, 0], motion_matrix[:, :, 1])
                mu, mv = mode(us.ravel())[0], mode(vs.ravel())[0]
                u, v = cv2.polarToCart(mu, mv)
                u, v = u[0][0], v[0][0]
            else:
                raise NotImplemented("Choose one of implemented modes: mean, trimmed_mean, median")

            accumulated_flow += np.array([u, v])
            transform_matrix = np.array([[1, 0, accumulated_flow[0]], [0, 1, accumulated_flow[1]]], dtype=np.float32)
            stabilized = cv2.warpAffine(im_frame, transform_matrix, (w, h))

            stab.append(cv2.cvtColor(stabilized, cv2.COLOR_BGR2RGB))

        reference_frame = im_frame

    filename = "_{}{}_{}".format(stab_mode, trimmed_mean_percentage if 'trimmed' in stab_mode else "", optical_flow_func)
    imageio.mimsave("stab/stab{}.gif".format(filename), stab)
    imageio.mimsave("stab/unstab.gif", unstab)
    print("yay")


def task_22(video_path, method="point"):

    if method == "point":
        point_matching(video_path)

    elif method == "fast":
        fast(video_path)

    else:
        raise NotImplementedError("Method must be 'point' of 'fast'")


# Implementation of J. Xu, H. Chang, S. Yang and M. Wang, "Fast feature-based video
# stabilization without accumulative global motion estimation," in IEEE Transactions on
# Consumer Electronics, vol. 58, no. 3, pp. 993-999, August 2012.

# Borrowed from https://github.com/francocurotto/Video-Stabilization

def fast(video_path):
    detector = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # parameters
    MATCH_THRES = float('Inf')
    RANSAC_THRES = 0.2
    BORDER_CUT = 10
    FILT = "gauss"
    FILT_WIDTH = 7
    FILT_SIGMA = 0.2
    FAST = True
    if FILT == "square":
        filt = (1.0/FILT_WIDTH) * np.ones(FILT_WIDTH)
    elif FILT == "gauss":
        filtx = np.linspace (-3*FILT_SIGMA, 3*FILT_SIGMA, FILT_WIDTH)
        filt = np.exp(-np.square(filtx) / (2*FILT_SIGMA))
        filt =  1/(np.sum(filt)) * filt

    videoArr = getVideoArray(video_path)

    trans = getTrans(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt, FAST)

    reconVideo(video_path, "off_the_shelf_fast.mp4", trans, BORDER_CUT)


def point_matching(video_path):

    cap = cv2.VideoCapture(video_path)


    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('off_the_shelf.mp4', fourcc, fps, (w, h))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in tqdm(range(n_frames-2)):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        _, curr = cap.read()

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)

        # traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray


    # smooth the trajectory
    trajectory = np.cumsum(transforms, axis=0)
    difference = smooth(trajectory, 5) - trajectory
    transforms_smooth = transforms + difference

    # apply transformation to sequence
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames-2):
        _, frame = cap.read()

        # get transformations

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i ,2]

        # build transfromation matrix
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_border(frame_stabilized)
        # frame_out = cv2.hconcat([frame, frame_stabilized])

        # frame_out = cv2.resize(frame_out, (w//4, h//4))
        out.write(frame_stabilized)


def task_31():
    gt_annot_file = 'datasets/ai_challenge_s03_c010-full_annotation.xml'
    frames_path = 'datasets/AICity_data/train/S03/c010/data'
    
    
    det_bb = read_detections_file("datasets/AICity_data/train/S03/c010/det/our_results_finetune_faster.txt")
    gt_bb = read_xml_gt_options(gt_annot_file, False, False)
    gt_bb = filter_gt(gt_bb, ["car"])
    
    video_n_frames = number_of_images_jpg(frames_path)
    det_bb_max_iou, idd = tracking_iou(frames_path, copy.deepcopy(det_bb), video_n_frames, mode='of')
    ap_max_iou = calculate_ap(det_bb_max_iou, gt_bb, 0, video_n_frames, mode='sort')
    
    print("Ap after tracking with maximum IoU: {}".format(ap_max_iou))
    
    det_bb_max_iou, idd = tracking_iou(frames_path, copy.deepcopy(det_bb), video_n_frames, mode='other')
    ap_max_iou = calculate_ap(det_bb_max_iou, copy.deepcopy(gt_bb), 0, video_n_frames, mode='sort')
    print("Ap after tracking with maximum IoU: {}".format(ap_max_iou))
    
    
    new_tracks = restore_tracks(frames_path, det_bb_max_iou)
    ap_max_iou = calculate_ap(new_tracks, copy.deepcopy(gt_bb), 0, video_n_frames, mode='sort')
    print("Ap after tracking with maximum IoU and of: {}".format(ap_max_iou))
    
    
        
    ini_frame = 700
    end_frame = 900
    animation_tracks(new_tracks, idd, ini_frame, end_frame, frames_path)
    

def compute_matching(block, space_search, error_function='sad'):
    block_size = block.shape
    window_size = space_search.shape




if __name__ == '__main__':
    # main()
    images_path = 'datasets/dummy/frames'

    modes = ['mean', 'trimmed_mean', 'median', 'mode']
    error_func = ['SSD', 'SAD']

    # for e_fun in error_func:
    #     for m in modes:
    #         if m == 'trimmed_mean':
    #             for trim in range(0, 5):
    #                 task_21(images_path, optical_flow_func=e_fun, stab_mode=m, trimmed_mean_percentage=trim / 10)
    #         else:
    #             task_21(images_path, optical_flow_func=e_fun, stab_mode=m)

    for m in modes:
        if m == 'trimmed_mean':
            for trim in range(0, 5):
                task_21(images_path, stab_mode=m, trimmed_mean_percentage=trim / 10)
        else:
            task_21(images_path, stab_mode=m)
