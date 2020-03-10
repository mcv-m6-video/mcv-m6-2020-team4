import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

from data import read_xml_gt_options
from data import save_frames, number_of_images_jpg, filter_gt
from utils.visualization import animate_iou, animation_2bb, plot_animation
from bgModels import bg_model, remove_bg, bg_estimation
from metrics.mAP import calculate_ap
from utils.visualization import animation_2bb
from glob import glob


def main():

    color_space = cv2.COLOR_BGR2GRAY
    # color_space = cv2.COLOR_BGR2HSV
    # color_space = cv2.COLOR_BGR2RGB
    channels = (0, 2)

    # print("Task 1")
    # task1("datasets/ai_challenge_s03_c010-full_annotation.xml", color_space=color_space)
    print("Task 2")
    task2(
        'datasets/AICity_data/train/S03/c010/data/',
        "datasets/ai_challenge_s03_c010-full_annotation.xml",
        color_space=color_space, channels=channels)
    print("Task 3")
    # task3()
    print("Task 4")


def task1(gt_path, color_space=cv2.COLOR_BGR2GRAY):
    path = 'datasets/AICity_data/train/S03/c010'
    save_frames(path)
    print("Finished saving")

    frames_path = 'datasets/AICity_data/train/S03/c010/data'
    video_n_frames = number_of_images_jpg(frames_path)

    # this is very time consuming, we should avoid comuting it more than once.
    mu, sigma = bg_model(frames_path, color_space)

    # Gaussian plot in the slides
    xx = np.linspace(0, 255, 1000)
    exp = 1 / (sigma[561, 712] * np.sqrt(2 * np.pi)) * \
        np.exp(-(xx - mu[561, 712]) ** 2 / (2 * sigma[561, 712] ** 2))
    plt.figure()
    plt.plot(xx, exp, color='blue', label='Distribution')
    plt.axvline(x=mu[561, 712], color='red', label='Mean')
    plt.axvline(x=mu[561, 712] - (sigma[561, 712] + 2), linestyle='--', color='green',
                label='Tolerance ' + r'$\alpha = 1$')
    plt.axvline(x=mu[561, 712] + (sigma[561, 712] + 2), linestyle='--', color='green')
    plt.axvline(x=mu[561, 712] - 3 * (sigma[561, 712] + 2), linestyle='--',
                color='purple', label='Tolerance ' + r'$\alpha = 5$')
    plt.axvline(x=mu[561, 712] + 3 * (sigma[561, 712] + 2),
                linestyle='--', color='purple')
    plt.legend()


    alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    aps7 = []

    # Test for different alpha values
    for a in alphas:
        det_bb = remove_bg(mu,
                           sigma,
                           a,
                           frames_path,
                           int(video_n_frames * 0.25),
                           video_n_frames,
                           animation=False,
                           denoise=True,
                           adaptive=False,
                           color_space=color_space)

        gt_bb = read_xml_gt_options(gt_path, True, True)

        ap = calculate_ap(det_bb, gt_bb, int(video_n_frames*0.25), video_n_frames, mode = 'area')
        animation_2bb('try_dnoise', '.gif', gt_bb, det_bb, frames_path, 10, 10, int(video_n_frames*0.25),
              int(1920 / 4), int(1080 / 4))


        print(a,ap)
        aps7.append(ap)

    plt.title('Median Filter')
#    plt.plot(alphas, aps3, label = 'Window size 3')
#    plt.plot(alphas, aps5, label = 'Window size 5')
    plt.plot(alphas, aps7, label = 'Window size 7')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('mAP')
    plt.legend()



def task2(frames_path, gt_path, color_space=cv2.COLOR_BGR2GRAY, channels=(0)):

    grid_search = False
    save_videos = False
    fine_tune_search = False
    videos_rgb_bb = True

    gt_bb = read_xml_gt_options(gt_path, True, True)

    classes_to_keep = ['car', 'bike']
    gt_bb = filter_gt(gt_bb, classes_to_keep)

    video_n_frames = number_of_images_jpg(frames_path)
    mu, sigma = bg_model(frames_path, color_space)

    if grid_search:
        mAPs = []
        alphas = [2, 2.5, 3, 3.5, 4, 4.5]
        rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for alpha in alphas:
            mAP_same_alfa = []
            for rho in rhos:
                det_bb = remove_bg(mu,
                                   sigma,
                                   alpha,
                                   frames_path,
                                   int(video_n_frames * 0.25),
                                   video_n_frames,
                                   color_space=color_space,
                                   adaptive=True,
                                   rho=rho, channels=channels)
                mAP = calculate_ap(
                    det_bb, gt_bb, int(
                        video_n_frames * 0.25), video_n_frames, mode='area')
                mAP_same_alfa.append(mAP)
                print(
                    "Alpha: {:2f} | Rho: {:2f} | mAP: {:2f} |".format(
                        alpha, rho, mAP))
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
        plt.savefig("_grid_search.png")
        plt.show()

    if save_videos:
        alpha = 3.5
        rho = 0.5
        det_bb = remove_bg(
            mu,
            sigma,
            alpha,
            frames_path,
            int(video_n_frames * 0.25) + 300,
            int(video_n_frames * 0.25) + 400,
            rho=rho,
            denoise=False,
            animation=True,
            color_space=color_space,
            adaptive=True,
            channels=channels)


        mAP = calculate_ap(
            det_bb, gt_bb, int(
                video_n_frames * 0.25), video_n_frames, mode='area')

        print(f"Channels: {channels}, Colorspace: {color_space}, mAP: {mAP}")

    if fine_tune_search:
        mAPs = []
        alphas = [3, 3, 3, 3.5, 3.5, 3.5, 4, 4, 4, 3.5, 3.5, 3.5, 3.5, 3.7, 3.2, 3.1, 3.3]
        rhos = [0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.1, 0.3, 0.15, 0.05, 0.03, 0.02, 0.01, 0.01]
        for i in range(0, len(alphas)):
            det_bb = remove_bg(mu,
                               sigma,
                               alphas[i],
                               frames_path,
                               int(video_n_frames * 0.25),
                               video_n_frames,
                               color_space=color_space,
                               adaptive=True,
                               rho=rhos[i],
                               denoise = True)
            mAP = calculate_ap(
                det_bb, gt_bb, int(
                    video_n_frames * 0.25), video_n_frames, mode='area')
            print(
                "Alpha: {:2f} | Rho: {:2f} | mAP: {:2f} |".format(
                    alphas[i], rhos[i], mAP))
            mAPs.append(mAP)

        fig, ax = plt.subplots()
        plt.xlim(2.9, 4.1)
        plt.ylim(-0.1, 0.5)
        ax.set_aspect(1)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Rho')

        for i in range(0, len(mAPs)):
            circle = plt.Circle((alphas[i], rhos[i]), mAPs[i]/25)
            ax.add_artist(circle)

        plt.savefig("fine_tune_search.png", bbox_inches='tight')
        plt.show()
    if videos_rgb_bb:
        alpha = 3.2
        rho = 0.02
        det_bb = remove_bg(mu,
                           sigma,
                           alpha,
                           frames_path,
                           int(video_n_frames * 0.25),
                           video_n_frames,
                           color_space=color_space,
                           adaptive=True,
                           rho=rho,
                           denoise = True)
        animation_2bb('rgb_bb_adaptive', '.gif', gt_bb, det_bb, frames_path, 10, 10, int(video_n_frames*0.25),
              int(1920 / 4), int(1080 / 4))
        alpha = 2.5
        det_bb = remove_bg(mu,
                           sigma,
                           alpha,
                           frames_path,
                           int(video_n_frames * 0.25),
                           video_n_frames,
                           color_space=color_space,
                           adaptive=False,
                           denoise = True)
        animation_2bb('rgb_bb_non_adaptive', '.gif', gt_bb, det_bb, frames_path, 10, 10, int(video_n_frames*0.25),
              int(1920 / 4), int(1080 / 4))

def task3():
    ims = sorted(glob("datasets/AICity_data/train/S03/c010/data/*.jpg"))

    gt_bb = read_xml_gt_options(
        'datasets/ai_challenge_s03_c010-full_annotation.xml', True, True)

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

                _, contours, _ = cv2.findContours(
                    fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
