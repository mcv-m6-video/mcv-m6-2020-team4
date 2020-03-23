import os
import cv2
import numpy as np

from data import save_frames, load_flow_data, process_flow_data
from optical_flow import OpticalFlowBlockMatching
from metrics.optical_flow import compute_optical_metrics
from utils.optical_flow_visualization import visualize_flow, flow_to_color, flow_to_hsv
from utils.cost_functions import compute_error
from utils.optical_flow_visualization import draw_optical_flow
from utils.utils import get_files_from_dir
from utils.visualization import visualize_3d_plot

def main():
    images_path = 'datasets/AICity_data/train/S03/c010/data'

    save_frames(images_path)
    print("Finished saving")

    print("Task 1.1")
    print("Computing flow of 000045_10:")
    flow_gt = "datasets/results/gt/000045_10.png"
    im1_path = "datasets/results/images/000045_10.png"
    im2_path = "datasets/results/images/000045_11.png"
    task11(images_path, flow_gt, im1_path, im2_path)
    print("Computing flow of 000157_10:")
    flow_gt = "datasets/results/gt/000157_10.png"
    im1_path = "datasets/results/images/000157_10.png"
    im2_path = "datasets/results/images/000157_11.png"
    task11(images_path, flow_gt, im1_path, im2_path)

def task11(frames_path, flow_gt, im1_path, im2_path):
    compute_optical_flow_metrics = False
    grid_search_block_area = True

    #Load ground truth
    gt = load_flow_data(flow_gt)
    gt = process_flow_data(gt)
    #Load frames
    first_frame = cv2.imread(im2_path)
    second_frame = cv2.imread(im2_path)

    if compute_optical_flow_metrics:
        #Compute optical flow
        flow_func = OpticalFlowBlockMatching(type="FW", block_size=9, area_search=40, error_function="SSD", window_stride=9)
        flow = flow_func.compute_optical_flow(first_frame, second_frame)

        #Compute metrics
        msen, psen = compute_optical_metrics(flow, gt, plot_error=True)
        print("MSEN: {}".format(msen))
        print("PSEN: {}".format(psen))

        #Visualize the computed optical flow
        print("visualize")
        visualize_flow(flow, simple=True)
        print("to color")
        flow_color = flow_to_color(flow[..., :2], convert_to_bgr=False)
        print("visualize")
        visualize_flow(flow_color)
        print("To hsv")
        hsv_flow = flow_to_hsv(flow)
        print("visualize hsv")
        visualize_flow(hsv_flow, hsv_format=True)

    if grid_search_block_area:
        area_sizes = np.array([20, 40, 60, 80, 100, 120])
        block_sizes = np.array([3,5,7,9,11,15,17,19,21])
        msens = np.zeros((len(block_sizes), len(area_sizes)))
        psens = np.zeros((len(block_sizes), len(area_sizes)))
        X,Y = np.meshgrid(area_sizes, block_sizes)
        for i, area_size in enumerate(area_sizes):
            for j, block_size in enumerate(block_sizes):
                #Compute optical flow
                flow_func = OpticalFlowBlockMatching(type="FW", block_size=block_size, area_search=area_size, error_function="SSD", window_stride=block_size)
                flow = flow_func.compute_optical_flow(first_frame, second_frame)
                msen, psen = compute_optical_metrics(flow, gt, plot_error=False)
                print("Area size: {:2f} | Block size: {:2f} | msen: {:2f} | psen: {:2f} |".format(area_size, block_size, msen, psen))
                msens[j,i] = msen
                psens[j,i] = psen
        # Plot msen grid search
        visualize_3d_plot(X, Y, msens, 'Area size', 'Block size', 'MSEN')
        # Plot psen grid search
        visualize_3d_plot(X, Y, psens, 'Area size', 'Block size', 'PSEN')

def task_21(frames_path):
    assert os.path.exists(frames_path)
    files = get_files_from_dir(frames_path, "jpg")
    assert len(files) != 0, "no frames in folder."



    block_size = 16
    search_window = 7

    accumulated_flow = np.zeros(2)
    from matplotlib import pyplot as plt

    for idx, frame in enumerate(sorted(files)):
        if idx == 0:
            im_frame = cv2.imread(frame, 0)
            h, w = im_frame.shape
            h, w = int(h / 4), int(w /4)
            im_frame = cv2.resize(im_frame, (w, h))
        else:
            reference_frame = im_frame
            im_frame = cv2.imread(frame, 0)
            h, w = im_frame.shape
            h, w = int(h / 4), int(w / 4)
            im_frame = cv2.resize(im_frame, (w, h))

            n_blocks_w = w // block_size
            n_blocks_h = h // block_size

            motion_matrix = np.zeros((n_blocks_h, n_blocks_w, 2))

            for i in range(n_blocks_h):
                for j in range(n_blocks_w):
                    block = im_frame[i * block_size:i * block_size + block_size, j * block_size:j * block_size + block_size]
                    i_init = max(i * block_size, 0)
                    j_init = max(i * block_size, 0)
                    i_end = min(i * block_size + block_size + search_window, h)
                    j_end = min(j * block_size + block_size + search_window, w)
                    space_search = reference_frame[i_init:i_end, j_init:j_end]

                    dx, dy = compute_matching(block, space_search)

                    motion_matrix[i, j, 0] = dx - (i - i_init)
                    motion_matrix[i, j, 1] = dy - (j - j_init)

            draw_optical_flow(im_frame, motion_matrix)

            accumulated_flow = motion_matrix.mean(axis=(0, 1))
            transform_matrix = np.array([[1, 0, accumulated_flow[0]], [0, 1, accumulated_flow[1]]])
            stabilized = cv2.warpAffine(im_frame, transform_matrix, (w, h))




def compute_matching(block, space_search, error_function='sad'):
    block_size = block.shape
    window_size = space_search.shape

    best_error = np.inf
    dx, dy = 0, 0

    for i in range(window_size[0] - block_size[0]):
        for j in range(window_size[1] - block_size[1]):
            space_block = space_search[i:i + block_size[0], j:j + block_size[1]]
            error = compute_error(block, space_block, error_function)
            if error < best_error:
                dx, dy = i, j

    return dx, dy


if __name__ == '__main__':
    main()
