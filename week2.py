import matplotlib.pylab as plt
import numpy as np

from bgModels import bg_model_grayscale, remove_bg, remove_adaptive_bg, bg_estimation
from data import read_xml_gt_options
from data import save_frames, number_of_images_jpg, filter_gt
from metrics.mAP import calculate_ap
from utils.visualization import animation_2bb
import cv2
from glob import glob

def main():
    print("Task 1")
    # task1("datasets/ai_challenge_s03_c010-full_annotation.xml")
    print("Task 2")
    task2('datasets/AICity_data/train/S03/c010/frames/', "datasets/ai_challenge_s03_c010-full_annotation.xml")
    print("Task 3")

    print("Task 4")


def task1(gt_path):
    path = 'datasets/AICity_data/train/S03/c010'
    save_frames(path)
    print("Finished saving")

    frames_path = 'datasets/AICity_data/train/S03/c010/data'
    video_n_frames = number_of_images_jpg(frames_path)

    # this is very time consuming, we should avoid comuting it more than once.
    mu, sigma = bg_model_grayscale(frames_path)

    # Gaussian plot in the slides
    xx = np.linspace(0, 255, 1000)
    exp = 1 / (sigma[561, 712] * np.sqrt(2 * np.pi)) * np.exp(-(xx - mu[561, 712]) ** 2 / (2 * sigma[561, 712] ** 2))
    plt.figure()
    plt.plot(xx, exp, color='blue', label='Distribution')
    plt.axvline(x=mu[561, 712], color='red', label='Mean')
    plt.axvline(x=mu[561, 712] - (sigma[561, 712] + 2), linestyle='--', color='green',
                label='Tolerance ' + r'$\alpha = 1$')
    plt.axvline(x=mu[561, 712] + (sigma[561, 712] + 2), linestyle='--', color='green')
    plt.axvline(x=mu[561, 712] - 3 * (sigma[561, 712] + 2), linestyle='--', color='purple',
                label='Tolerance ' + r'$\alpha = 5$')
    plt.axvline(x=mu[561, 712] + 3 * (sigma[561, 712] + 2), linestyle='--', color='purple')
    plt.legend()

    alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    aps7 = []

    # Test for different alpha values
    for a in alphas:
        det_bb = remove_bg(mu, sigma, a, frames_path, int(video_n_frames * 0.25), video_n_frames,
                           animation=False, denoise=True)

        gt_bb = read_xml_gt_options(gt_path, True, True)

        ap = calculate_ap(det_bb, gt_bb, int(video_n_frames * 0.25), video_n_frames, mode='area')
        print(a, ap)
        aps7.append(ap)

    #
    #
    animation_2bb('try', '.gif', gt_bb, det_bb, frames_path, 10, 100, int(video_n_frames * 0.25) + 1,
                  int(1920 / 4), int(1080 / 4))


#
#
# animation_2bb('try', '.gif', gt_bb, det_bb, frames_path, 10, 100, int(video_n_frames*0.25) + 1,
#              int(1920 / 4), int(1080 / 4))


def task2(frames_path, gt_path):
    grid_search = False
    save_videos = True
    mu, sigma = bg_model_grayscale(frames_path)
    np.savetxt("mu.csv", mu, delimiter=",")
    np.savetxt("sigma.csv", sigma, delimiter=",")
    gt_bb = read_xml_gt_options(gt_path, True, True)

    classes_to_keep = ['car', 'bike']
    gt_bb = filter_gt(gt_bb, classes_to_keep)

    mu = np.genfromtxt("mu.csv", delimiter=',')
    sigma = np.genfromtxt("sigma.csv", delimiter=',')
    video_n_frames = number_of_images_jpg(frames_path)

    if grid_search:
        mAPs = []
        alphas = [2, 2.5, 3, 3.5, 4]
        rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for alpha in alphas:
            mAP_same_alfa = []
            for rho in rhos:
                det_bb = remove_adaptive_bg(mu, sigma, alpha, rho, frames_path, int(video_n_frames * 0.25),
                                            video_n_frames)
                mAP = calculate_ap(det_bb, gt_bb, int(video_n_frames * 0.25), video_n_frames, mode='area')
                mAP_same_alfa.append(mAP)
                print("Alpha: {:2f} | Rho: {:2f} | mAP: {:2f} |".format(alpha, rho, mAP))
            mAPs.append(mAP_same_alfa)

        # Plot the surface
        X, Y = np.meshgrid(alphas, rhos)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, mAPs, rstride=0.1, cstride=0.1,
                               cmap='viridis', edgecolor='none')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Rho')
        ax.set_zlabel('mAP')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("adaptive_grid_search.png")
        plt.show()
    if save_videos:
        alpha = 3.5
        rho = 0.2
        det_bb = remove_adaptive_bg(mu, sigma, alpha, rho, frames_path, 700, 800, animation=True)


def task3():
    ims = sorted(glob("datasets/AICity_data/train/S03/c010/frames/*.jpeg"))

    gt_bb = read_xml_gt_options('datasets/ai_challenge_s03_c010-full_annotation.xml', True, True)

    init_percent = .25
    init_images = int(len(ims) * init_percent)
    first_eval_image = init_images + 1

    modes = ['mog', 'knn', 'gmg']
    model_params = {
        'mog': {
            'varThreshold': 32
        },
        'knn': {
            'dist2Threshold': 350
        },
        'gmg': {
            'initializationFrames': init_images
        }

    }

    for mode in modes:
        bg_sub = bg_estimation(mode, **model_params[mode])
        det_bb = []
        for i, im in enumerate(ims):
            im = cv2.imread(im)
            fg = bg_sub.apply(im)
            # cv2.imshow("a", fg)
            # cv2.waitKey(0)

            if i >= first_eval_image:

                contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    # Open cv documentation:  (x,y) be the top-left coordinate
                    # of the rectangle and (w,h) be its width and height.
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if w > 10 and h > 10:
                        det_bb.append([i, 'car', 0, x, y, x + w, y + h])

        ap = calculate_ap(det_bb, gt_bb, init_images + 1, len(ims), mode='area')
        print("{} - {}".format(mode, ap))


if __name__ == '__main__':
    main()
