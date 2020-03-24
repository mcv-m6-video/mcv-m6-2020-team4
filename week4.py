import os

import cv2
import imageio
import numpy as np
from scipy.stats import trim_mean, mode

from data import save_frames, load_flow_data, process_flow_data
from metrics.optical_flow import compute_optical_metrics
from optical_flow import OpticalFlowBlockMatching
from utils.optical_flow_visualization import visualize_flow, flow_to_hsv, visualize_flow_v2
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
    # task11(images_path, flow_gt, im1_path, im2_path)


def task11(frames_path, flow_gt, im1_path, im2_path, ):
    compute_optical_flow_metrics = True
    grid_search_block_area = False

    # Load ground truth
    gt = load_flow_data(flow_gt)
    gt = process_flow_data(gt)
    """
    print(gt.shape)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i,j,2]!=0.:
                print((i,j))
                print(gt[i,j,:])
    """
    # Load frames
    first_frame = cv2.imread(im1_path)
    second_frame = cv2.imread(im2_path)

    if compute_optical_flow_metrics:
        # Compute optical flow
        flow_func = OpticalFlowBlockMatching(type="FW", block_size=9, area_search=40, error_function="SSD",
                                             window_stride=9)
        flow = flow_func.compute_optical_flow(first_frame, second_frame)

        # Compute metrics
        msen, psen = compute_optical_metrics(flow, gt, plot_error=False)
        print("MSEN: {}".format(msen))
        print("PSEN: {}".format(psen))

        # Visualize the computed optical flow
        visualize_flow_v2(first_frame, flow)
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
