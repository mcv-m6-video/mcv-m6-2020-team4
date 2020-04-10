import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


# Taken from https://github.com/tomrunia/OpticalFlow_Visualization

# S. Baker, D. Scharstein, J. Lewis, S. Roth, M. J. Black, and R. Szeliski.
# A database and evaluation methodology for optical flow.
# In Proc. IEEE International Conference on Computer Vision (ICCV), 2007.

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    """

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    """

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


def flow_to_hsv(flow):
    hsv = np.zeros(flow.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    ang = np.arctan2(flow[...,1], flow[...,0]) + np.pi
    hsv[..., 0] = cv2.normalize(ang, None, 0, 179, cv2.NORM_MINMAX)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return hsv


def visualize_flow(flow, suffix="", hsv_format=False, simple=False):
    if hsv_format:
        rgb = cv2.cvtColor(flow, cv2.COLOR_HSV2RGB)
        cv2.imshow("Angle" + suffix, flow[..., 0])
        cv2.imshow("Magnitude" + suffix, flow[..., 2])
        cv2.imwrite("cv.png", rgb)
        

    elif simple:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = mag[::5, ::5]
        ang = ang[::5, ::5]
        plt.figure()
        plt.title("Flow" + suffix)
        plt.quiver(mag * np.cos(ang), mag * np.sin(ang))
        plt.show()

    else:
        cv2.imshow("Flow" + suffix, flow)

    cv2.waitKey(0)


def visualize_flow_v2(im, flow):
    """
    Read of_image with load_flow_data
    """
    step = 30
    w, h = im.shape[:2]
    x, y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    u = flow[::step, ::step, 0]
    v = flow[::step, ::step, 1]
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title("Flow")
    plt.quiver(y, x, (u - np.mean(u)).T, -(v - np.mean(v)).T, color='red')
    plt.show()


def draw_optical_flow(image, motion_vector):
    step = image.shape[0] // motion_vector.shape[0], image.shape[1] // motion_vector.shape[1]
    if step == (1, 1):
        step = (10, 10)
    w, h = image.shape[:2]
    x, y = np.meshgrid(np.arange(0, w, step[0]), np.arange(0, h, step[1]))
    u = motion_vector[:, :, 0]
    v = motion_vector[:, :, 1]
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("Flow")
    plt.quiver(y, x, (u - np.mean(u)).T, -(v - np.mean(v)).T, color='red')
    plt.show()
