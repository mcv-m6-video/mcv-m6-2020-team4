import os

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from torch import nn
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from utils.data import read_gt_txt


def get_crops_from_dets(image_folder, dets, out_folder, camera_id):
    frames = np.unique(dets[:, 0])
    frame_dets = {int(f): dets[dets[:, 0] == f] for f in frames}
    for frame, dets in frame_dets.items():
        im = cv2.imread(os.path.join(image_folder, "frame_{:04d}.jpg".format(frame)))
        for det in dets:
            det_box = det[-4:].astype(float).astype(int)
            cutted = im[det_box[1]:det_box[3], det_box[0]:det_box[2], :]
            cutted = cv2.resize(cutted, (224, 224))
            os.makedirs(os.path.join(out_folder, det[2]), exist_ok=True)
            cv2.imwrite(os.path.join(out_folder, det[2], "{}_{}.jpg".format(camera_id, det[0])), cutted)


class Net(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        in_features = self.feature_extractor.fc.in_features
        self.embedding_size = embedding_size
        self.feature_extractor.fc = nn.Linear(in_features, embedding_size)

    def forward(self, *input, **kwargs):
        return self.feature_extractor(input[0])

    def get_embedding(self, image):
        with torch.no_grad():
            out = self.feature_extractor(image[None, ...])
            return out


def sync_frames(root_dir, cameras, offsets):
    synced_frames = {}
    camera_frames = {}

    for camera, offset in zip(cameras, offsets):
        if camera == 'c015':
            fps = 8
        else:
            fps = 10
        frames = sorted(os.listdir(os.path.join(root_dir, camera, "images")))
        camera_frames[camera] = frames
        for i, frame in enumerate(frames):
            if i / fps >= offset:
                if i in synced_frames.keys():
                    synced_frames[i].append(camera)
                else:
                    synced_frames[i] = [camera]

    return synced_frames, camera_frames


def cut_det(im, box):
    cutted = im[box[1]: box[3], box[0]:box[2], :]
    cutted = cv2.resize(cutted, (224, 224))
    cutted = T.ToTensor()(cutted)

    return cutted


def get_detector():
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    return DefaultPredictor(cfg)


def perform_det(predictor, im):
    preds = predictor(im)
    boxes = preds['instances'].get_fields()['pred_boxes']
    scores = preds['instances'].get_fields()['scores']
    classes = preds['instances'].get_fields()['pred_classes']

    boxes = boxes[classes == 2].tensor.cpu().numpy()
    scores = scores[classes == 2].cpu().numpy()

    return boxes


def generate_car_crops(root_dir, cameras):
    for camera in cameras:
        dets = np.array(read_gt_txt(os.path.join(root_dir, camera, "gt", "gt.txt")))
        out_dir = os.path.join(root_dir, "crops")
        os.makedirs(out_dir, exist_ok=True)
        get_crops_from_dets(os.path.join(root_dir, camera, "images"), dets, out_dir, camera)


if __name__ == '__main__':
    root_dir = "/home/devsodin/Downloads/AIC20_track3_MTMC/AIC20_track3/train/S04"
    cameras = os.listdir(root_dir)
    # generate_car_crops(root_dir, cameras)

    train = ImageFolder("/home/devsodin/Downloads/AIC20_track3_MTMC/AIC20_track3/data/train")
    test = ImageFolder("/home/devsodin/Downloads/AIC20_track3_MTMC/AIC20_track3/data/test")
