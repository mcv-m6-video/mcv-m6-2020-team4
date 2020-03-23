import os

import cv2
import numpy as np

from utils.cost_functions import compute_error
from utils.optical_flow_visualization import draw_optical_flow
from utils.utils import get_files_from_dir


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
    task_21("datasets/dummy/frames")
