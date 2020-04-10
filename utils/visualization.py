import cv2
import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from utils.data import add_noise_to_bbox
from metrics.iou import bbox_iou


def plot_animation(x_elements, y_elements, x_label, y_label, y_label_range, frame_rate):
    """
    Creates an animation plot of the data x_elements and y_elements with the
    indicated frame_rate
    """
    delay = 1000 / frame_rate
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim(y_label_range)
    line, = ax1.plot(x_elements, y_elements)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ani = animation.FuncAnimation(fig, update_plot_animation, len(x_elements), fargs=[x_elements, y_elements, line],
                                  interval=delay)
    return ani


def update_plot_animation(num, x, y, line):
    """
    Updates the plot animation
    """
    line.set_data(x[:num], y[:num])
    return line,


def animate_iou(dict_frame_gt_bb):
    """
    Given a dictionary with frames and ground truth bounding boxes computes mean
    IoU between the bounding boxes of each frame and the ground truth bounding
    boxes with a gussian noise added.
    Returns an animation of the evolution of mean IoU over time
    """
    ious = []
    for key, value in sorted(dict_frame_gt_bb.items(), key=lambda x: int(x[0])):
        value = [v[:-1] for v in value]
        noisy_bb = add_noise_to_bbox(value, -5, 5)
        iou = 0.0
        for i, gt_bb in enumerate(value):
            iou += bbox_iou([float(i) for i in gt_bb], noisy_bb[i])
        ious.append(iou / len(value))
    frames = list(range(0, len(ious)))
    animation = plot_animation(frames, ious, "Frame", "IoU", [0, 1], 10)
    return animation


def animation_2bb(name, format, gt_bb, bb_cords, frame_path, fps=10, seconds=10, ini=0, width=480, height=270):
    """
    This function records a video of some frames with both the GT (green)
    and detection (blue) bounding boxes. If we have a confidence value that number
    is added on top of the bounding box.
    Input
        Name: Name of the file to save
        format: format of the file, it can be .avi or .gif (. must be included)
        gt_bb: ground truth bounding boxes in the same format as reed
        bb_cords: bounding box for the detection
    """

    # in case we have a confidence value in the detection
    if len(bb_cords) == 7:
        confid = True
    else:
        confid = False

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./' + name + format, fourcc, float(fps), (width, height))

    lst_gt = [item[0] for item in gt_bb]
    lst_nogt = [item[0] for item in bb_cords]
    images = []

    for i in range(fps * seconds):
        f_val = i + ini
        frame1 = cv2.imread((frame_path + '/frame_{:04d}.jpg').format(f_val))

        args_gt = [i for i, num in enumerate(lst_gt) if num == f_val]
        for ar in args_gt:
            # Ground truth bounding box in green
            cv2.rectangle(frame1, (int(gt_bb[ar][3]), int(gt_bb[ar][4])),
                          (int(gt_bb[ar][5]), int(gt_bb[ar][6])), (0, 255, 0), 2)

        args_nogt = [i for i, num in enumerate(lst_nogt) if num == f_val]
        for ar in args_nogt:
            # guessed GT in blue
            cv2.rectangle(frame1, (int(bb_cords[ar][3]), int(bb_cords[ar][4])),
                          (int(bb_cords[ar][5]), int(bb_cords[ar][6])), (255, 0, 0), 2)

            if confid:
                cv2.putText(frame1, str(bb_cords[ar][6]) + " %",
                            (int(bb_cords[i][2]), int(bb_cords[i][3]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        frame1 = cv2.resize(frame1, (width, height))

        if format == '.gif':
            images.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        else:
            video.write(frame1)

    if format == '.gif':
        imageio.mimsave(name + format, images)
    else:
        video.release()


def frame_with_2bb(gt_bb, det_bb, frame_path, f_val):
    lst_gt = [item[0] for item in gt_bb]
    lst_nogt = [item[0] for item in det_bb]
    frame1 = cv2.imread((frame_path + '/frame_{:04d}.jpg').format(f_val+1))

    args_gt = [i for i, num in enumerate(lst_gt) if num == f_val]

    np.random.seed(34)
    r = np.random.randint(0, 256, len(args_gt), dtype = int)
    g = np.random.randint(0, 256, len(args_gt), dtype = int)
    b = np.random.randint(0, 256, len(args_gt), dtype = int)

#    for ar in args_gt:
#        # Ground truth bounding box in green
#        cv2.rectangle(frame1, (int(gt_bb[ar][3]), int(gt_bb[ar][4])),
#                      (int(gt_bb[ar][5]), int(gt_bb[ar][6])), (0, 255, 0), 2)

    args_nogt = [i for i, num in enumerate(lst_nogt) if num == f_val]
    for i, ar in enumerate(args_nogt):
        # guessed GT in blue
        cv2.rectangle(frame1, (int(det_bb[ar][3]), int(det_bb[ar][4])),
                      (int(det_bb[ar][5]), int(det_bb[ar][6])),
                      (int(r[i]),int(g[i]),int(b[i])), 2)

#    frame1 = cv2.resize(frame1, (int(1920 / 4), int(1080 / 4)))
#    frame1 = cv2.resize(frame1, (int(1920), int(1080)))
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
#    imageio.imsave('frame{}_with_2bb.png'.format(f_val), frame1)
    return frame1

def frames_to_gif(filename, frames):
    frames = frames.astype('uint8')
    imageio.mimsave(filename, frames)



def animation_tracks(det_bb, idd, ini_frame, end_frame, frames_path):
    """
    This function creates an animation of the tracks assigning a different color
    to eack track. It also draws a number because the gif animation seems to be
    changing color for the same track when it is not.
    """
    np.random.seed(34)
    r = np.random.randint(0, 256, idd, dtype = int)
    g = np.random.randint(0, 256, idd, dtype = int)
    b = np.random.randint(0, 256, idd, dtype = int)

    images = []

    lst_bb = [item[0] for item in det_bb]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for f_val in range(ini_frame, end_frame):
        frame_bb = [det_bb[i] for i, num in enumerate(lst_bb) if num == f_val]
        frame1 = cv2.imread((frames_path + '/frame_{:04d}.jpg').format(f_val))
        for fr in frame_bb:
            cv2.rectangle(frame1, (int(fr[3]), int(fr[4])),
                  (int(fr[5]), int(fr[6])),
                  (int(r[fr[2]]), int(g[fr[2]]), int(b[fr[2]])), 2)
            cv2.putText(frame1, str(fr[2]), (int(fr[3]),
                        int(fr[4]) - 10), font, 0.75,
                        (int(r[fr[2]]), int(g[fr[2]]), int(b[fr[2]])), 2, cv2.LINE_AA)
#        frame1 = cv2.resize(frame1, (int(1920 / 2), int(1080 / 2)))
        images.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
#    imageio.mimsave('tracking4.gif', images)
    imageio.mimsave('track_lala.tiff', images)

def visualize_3d_plot(X, Y, Z, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y,  Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
