import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='M6-Project')
    parser.add_argument('--channels', default=0, nargs="*", type=int, help='channels to select')
    parser.add_argument('--color', default="gray", type=str, choices=["rgb", "gray", "ycrcb", "hsv", "lab"], help='colorspace to use')

    return parser.parse_args()



