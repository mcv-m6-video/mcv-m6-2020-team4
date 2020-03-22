import cv2
from data import save_frames, load_flow_data, process_flow_data
from optical_flow import OpticalFlowBlockMatching
from metrics.optical_flow import compute_optical_metrics
from utils.optical_flow_visualization import visualize_flow, flow_to_color, flow_to_hsv

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


if __name__ == '__main__':
    main()
