import numpy as np
from matplotlib import pyplot as plt

def compute_optical_metrics(prediction, gt, thr=3, plot_error=False):
    # discard occlusions
    occ = gt[:, :, 2] != 0

    diff = ((gt[..., :2] - prediction[..., :2]) ** 2)[occ]
    error = np.sqrt(diff[:, 0] + diff[:, 1])
    if plot_error:
        plt.hist(error)
        plt.show()
    msen = error.mean()
    psen = (error > thr).sum() / error.size * 100

    return msen, psen