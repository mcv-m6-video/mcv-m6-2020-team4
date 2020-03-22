import os
import cv2
import numpy as np
from data import save_frames, load_flow_data, process_flow_data
from optical_flow import OpticalFlowBlockMatching
from metrics.optical_flow import compute_optical_metrics
from utils.optical_flow_visualization import visualize_flow, flow_to_color, flow_to_hsv
from utils.cost_functions import compute_error
from utils.utils import get_files_from_dir


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
    #Load ground truth
    gt = load_flow_data(flow_gt)
    gt = process_flow_data(gt)

    #Compute optical flow
    first_frame = cv2.imread(im2_path)
    print(first_frame.shape)
    second_frame = cv2.imread(im2_path)
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

def task_21(frames_path):
    assert os.path.exists(frames_path)
    files = get_files_from_dir(frames_path, "jpg")
    assert len(files) != 0, "no frames in folder."

    previous_frame = cv2.imread(files[0], 0)
    frames = files[1:]

    block_size = 16
    search_window = 7

    motions = []

    for frame in frames:
        im_frame = cv2.imread(frame, 0)
        h, w = im_frame.shape
        # ref_image = np.zeros((h + 2 * search_window, w + 2 * search_window), 'uint8')
        # ref_image[search_window:h + search_window, search_window: w + search_window] = previous_frame
        ref_image  = previous_frame

        # cv2.imshow("im", ref_image)
        # cv2.waitKey(0)
        n_blocks_w = w // block_size
        n_blocks_h = h // block_size

        motion_matrix = np.zeros((n_blocks_h, n_blocks_w, 2))

        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = im_frame[i * block_size:i * block_size + block_size, j * block_size:j * block_size + block_size]
                i_init = max(i *block_size, 0)
                j_init = max(i * block_size, 0)
                i_end = min(i*block_size + block_size + search_window, h)
                j_end = min(j*block_size + block_size + search_window, w)
                space_search = ref_image[i_init:i_end, j_init:j_end]

                dx, dy = compute_matching(block, space_search)

                motion_matrix[i, j, 0] = dx - (i - i_init)
                motion_matrix[i, j, 1] = dy - (j - j_init)

        motions.append(motion_matrix)
        previous_frame = im_frame
        print(motion_matrix)

    print(motions)


def compute_matching(block, space_search, error_function='sad'):
    block_size = block.shape
    window_size = space_search.shape

    best_error = np.inf
    dx, dy = 0, 0

    for i in range(window_size[0] - block_size[0]):
        for j in range(window_size[1] - block_size[1]):
            space_block = space_search[i:i + block_size[0], j:j + block_size[1]]
            error = compute_error(block, space_block, error_function)
            if error <= best_error:
                dx, dy = i, j

    return dx, dy


if __name__ == '__main__':
    main()
