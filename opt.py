import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='M6-Project')
    parser.add_argument('--channels', default=0, nargs="*", type=int, help='channels to select')
    parser.add_argument('--color', default="gray", type=str, choices=["rgb", "gray", "ycrcb", "hsv", "lab"], help='colorspace to use')

    return parser.parse_args()

def parse_args_week5():

    parser = argparse.ArgumentParser(description='M6-Project-Week5')
    parser.add_argument('--task', default=1, type=int, choices=[1, 2], help='Task to execute')
    parser.add_argument('--detector', default="MaskR-CNN", type=str, choices=["MaskR-CNN", "YOLO", "SSD"], help='Detector to obtain the car detections')
    parser.add_argument('--trackingMethod', default="MaxOverlap", type=str, choices=["MaxOverlap", "Kalman"], help='Method used for the tracking')
    parser.add_argument('--postprocessing', default="None", type=str, choices=["None", "RemoveParked"], help='Postprocessing applied after the tracking')
    parser.add_argument('--visualization', default=0, type=int, choices=[0, 1], help='Store animation of the tracking')

    return parser.parse_args()
