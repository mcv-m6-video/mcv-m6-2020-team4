import cv2
import matplotlib.pylab as plt
import numpy as np

from bgModels import bg_model, remove_bg, bg_estimation, fg_segmentation_to_boxes
from data import read_xml_gt_options
from data import save_frames, number_of_images_jpg, filter_gt
from metrics.mAP import calculate_ap
from opt import parse_args
from utils.visualization import animation_2bb
from glob import glob
from utils.utils import get_files_from_dir
from utils.visualization import animation_2bb, frames_to_gif



def main():

    opt = parse_args()
    print(opt.__dict__)

    if opt.color == "gray":
        color_space = cv2.COLOR_BGR2GRAY
    elif opt.color == "hsv":
        color_space = cv2.COLOR_BGR2HSV
    elif opt.color == "rgb":
        color_space = cv2.COLOR_BGR2RGB
    elif opt.color == "ycrcb":
        color_space = cv2.COLOR_BGR2YCrCb
    elif opt.color == "lab":
        color_space = cv2.COLOR_BGR2LAB

    channels = tuple(opt.channels)

    # print("Task 1")
    # task1("datasets/ai_challenge_s03_c010-full_annotation.xml", color_space=color_space)
    print("Task 2")
    task2(
        'datasets/AICity_data/train/S03/c010/data/',
        "datasets/ai_challenge_s03_c010-full_annotation.xml",
        color_space=color_space, channels=channels)

    # print("Task 3")
    # task3("datasets/AICity_data/train/S03/c010/data", 'datasets/ai_challenge_s03_c010-full_annotation.xml',
    #       save_to_disk=True)
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

        ap = calculate_ap(det_bb, gt_bb, int(video_n_frames * 0.25), video_n_frames, mode='area')
        animation_2bb('try_dnoise', '.gif', gt_bb, det_bb, frames_path, 10, 10, int(video_n_frames * 0.25),
                      int(1920 / 4), int(1080 / 4))

        print(a, ap)
        aps7.append(ap)

    plt.title('Median Filter')
    #    plt.plot(alphas, aps3, label = 'Window size 3')
    #    plt.plot(alphas, aps5, label = 'Window size 5')
    plt.plot(alphas, aps7, label='Window size 7')
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
        alpha = 3.2
        rho = 0.02
        det_bb = remove_bg(
            mu,
            sigma,
            alpha,
            frames_path,
            800,
            900,
            rho=rho,
            denoise=True,
            animation=True,
            color_space=color_space,
            adaptive=True,
            channels=channels)


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
                               denoise=True)
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
            circle = plt.Circle((alphas[i], rhos[i]), mAPs[i] / 25)
            ax.add_artist(circle)

        plt.savefig("fine_tune_search.png", bbox_inches='tight')
        plt.show()
    if videos_rgb_bb:
        alpha = 2.5
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
                           denoise=True,
                           channels=channels)
        print("Calculating AP")
        mAP = calculate_ap(
            det_bb, gt_bb, int(
                video_n_frames * 0.25), video_n_frames, mode='area')
        animation_2bb('rgb_bb_adaptive', '.gif', gt_bb, det_bb, frames_path, 10, 10, 800,
              int(1920 / 4), int(1080 / 4))
        print(f"Channels: {channels}, Colorspace: {color_space}, mAP: {mAP}")
        # alpha = 2.5
        # det_bb = remove_bg(mu,
        #                    sigma,
        #                    alpha,
        #                    frames_path,
        #                    int(video_n_frames * 0.25),
        #                    video_n_frames,
        #                    color_space=color_space,
        #                    adaptive=False,
        #                    denoise = True)

        # print("Calculating AP")
        # mAP = calculate_ap(
        #     det_bb, gt_bb, int(
        #         video_n_frames * 0.25), video_n_frames, mode='area')

        # animation_2bb('rgb_bb_non_adaptive', '.gif', gt_bb, det_bb, frames_path, 10, 10, int(video_n_frames*0.25),
        #       int(1920 / 4), int(1080 / 4))

def task3(frames_path, annots_file, estimation_percent=.25, save_to_disk=False, save_size=(480, 270)):
    ims = get_files_from_dir(frames_path, "jpg")
    gt_bb = read_xml_gt_options(annots_file, True, True)

    init_images = int(len(ims) * estimation_percent)
    first_eval_image = init_images + 1
    modes = ['mog', 'knn', 'gmg', 'LSBP']
    model_params = {
        'mog': {
            # 'varThreshold': 32
        },
        'knn': {
            # 'dist2Threshold': 350
        },
        'gmg': {
            # 'initializationFrames': init_images
        },
        'LSBP': {

        },

    }

    for mode in modes:
        bg_sub = bg_estimation(mode, **model_params[mode])
        det_bb = []
        if save_to_disk:
            frames = np.zeros((len(ims) - first_eval_image, save_size[1], save_size[0]))

        for i, im in enumerate(ims):
            im = cv2.imread(im)
            fg = bg_sub.apply(im)
            fg[fg != 255] = 0

            if i >= first_eval_image:
                det_bb += fg_segmentation_to_boxes(fg, i)

            if save_to_disk:
                fg = cv2.resize(fg, save_size)
                frames[i - first_eval_image, ...] = fg

        ap = calculate_ap(det_bb, gt_bb, init_images + 1, len(ims), mode='area')
        if save_to_disk:
            animation_2bb('{}_bg'.format(mode), '.gif', gt_bb, det_bb, frames_path, ini=first_eval_image)
            frames_to_gif("fg_{}.gif".format(mode), frames)

        print("{} - {}".format(mode, ap))


if __name__ == '__main__':
    main()
