import numpy as np


def compute_optical_metrics(prediction, gt, thr=3):
    # discard occlusions
    occ = gt[:, :, 2] != 0

    diff = ((gt[..., :2] - prediction[..., :2]) ** 2)[occ]
    error = np.sqrt(diff[:, 0] + diff[:, 1])
    msen = error.mean()
    psen = (error > thr).sum() / error.size * 100

    return msen, psen