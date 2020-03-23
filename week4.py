import os

import cv2
import numpy as np

from data import save_frames, load_flow_data, process_flow_data
from metrics.optical_flow import compute_optical_metrics
from optical_flow import OpticalFlowBlockMatching
from utils.optical_flow_visualization import visualize_flow, flow_to_color, flow_to_hsv, draw_optical_flow
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

    # Load ground truth
    gt = load_flow_data(flow_gt)
    gt = process_flow_data(gt)
    # Load frames
    first_frame = cv2.imread(im2_path)
    second_frame = cv2.imread(im2_path)

    if compute_optical_flow_metrics:
        # Compute optical flow
        flow_func = OpticalFlowBlockMatching(type="FW", block_size=9, area_search=40, error_function="SSD",
                                             window_stride=9)
        flow = flow_func.compute_optical_flow(first_frame, second_frame)

        # Compute metrics
        msen, psen = compute_optical_metrics(flow, gt, plot_error=True)
        print("MSEN: {}".format(msen))
        print("PSEN: {}".format(psen))

        # Visualize the computed optical flow
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
        visualize_3d_plot(X, Y, psens, 'Area size', 'Block size', 'PSEN')


def task_21(frames_path):
    assert os.path.exists(frames_path)
    files = get_files_from_dir(frames_path, "jpg")
    assert len(files) != 0, "no frames in folder."

    opt_flow = OpticalFlowBlockMatching(type="BW", block_size=16, area_search=16, error_function="SAD",
                                        window_stride=16)

    block_size = 16
    search_window = 7

    accumulated_flow = np.zeros(2)
    stab = []
    unstab = []
    for idx, frame in enumerate(sorted(files)):
        if idx == 0:
            im_frame = cv2.imread(frame)
            h, w = im_frame.shape[:2]
            h, w = int(h / 4), int(w / 4)
            im_frame = cv2.resize(im_frame, (w, h))
            stab.append(im_frame)
            unstab.append(im_frame)
        else:
            reference_frame = im_frame
            im_frame = cv2.imread(frame)
            h, w = im_frame.shape[:2]
            h, w = int(h / 4), int(w / 4)
            im_frame = cv2.resize(im_frame, (w, h))
            unstab.append(im_frame)

            motion_matrix = opt_flow.compute_optical_flow(reference_frame, im_frame)
            draw_optical_flow(im_frame, motion_matrix[:,:,:2])

            from scipy.stats import trim_mean, mode
            # u = trim_mean(motion_matrix[:, :, 0], 0.1, axis=None)
            # v = trim_mean(motion_matrix[:, :, 1], 0.1, axis=None)
            # u, v = motion_matrix[:,:, 0].mean(axis=(0,1)), motion_matrix[:,:, 1].mean(axis=(0,1))
            motion_matrix = motion_matrix[motion_matrix[:, : ,-1] == 1]
            u,v = mode(motion_matrix[:, 0],axis=None)[0], mode(motion_matrix[:, 1],axis=None)[0]

            accumulated_flow += np.array([u[0], v[0]])
            transform_matrix = np.array([[1, 0, accumulated_flow[0]], [0, 1, accumulated_flow[1]]], dtype=np.float32)
            stabilized = cv2.warpAffine(im_frame, transform_matrix, (w, h))

            stab.append(stabilized)
    import imageio

    imageio.mimsave("stab.gif", stab)
    imageio.mimsave("unstab.gif", unstab)


if __name__ == '__main__':
    images_path = 'datasets/dummy/frames'
    task_21(images_path)
