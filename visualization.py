import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from data import load_annots, filter_annots, add_gauss_noise_to_bboxes
from metrics import bbox_iou

def plot_animation(x_elements, y_elements, x_label, y_label, y_label_range, frame_rate):
    """
    Creates an animation plot of the data x_elements and y_elements with the
    indicated frame_rate
    """
    delay = 1000 / frame_rate
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_ylim(y_label_range)
    line, = ax1.plot(x_elements, y_elements)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ani = animation.FuncAnimation(fig, update_plot_animation, len(x_elements), fargs=[x_elements, y_elements, line],interval=delay)
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
    for key,value in sorted(dict_frame_gt_bb.items(), key=lambda x: int(x[0])):
        noisy_bb = add_gauss_noise_to_bboxes(value, 5)
        iou = 0.0
        for i, gt_bb in enumerate(value):
            iou += bbox_iou([float(i) for i in gt_bb], noisy_bb[i])
        ious.append(iou/len(value))
    frames = list(range(0, len(ious)))
    animation = plot_animation(frames, ious,"Frame", "IoU", [0,1], 10)
    return animation

if __name__ == '__main__':
    annots = load_annots("datasets/ai_challenge_s03_c010-full_annotation.xml")

    classes = ['car', ]
    annots = filter_annots(annots, classes=classes)
    animation = animate_iou(annots)
    animation.save('test.gif')
