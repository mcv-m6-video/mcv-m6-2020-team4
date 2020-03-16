import os

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode

from data import filter_gt, read_xml_gt
from metrics.mAP import calculate_ap
from utils.visualization import animation_2bb

thing_classes = {
    'car': 0,

}


def get_AICity_dataset(image_path, annot_file):
    assert os.path.exists(annot_file)

    f = open(annot_file, 'r')
    dataset = []

    annots = [l.split(",") for l in f.readlines()]

    annots = np.array(annots)

    sequence_frames = annots[:, 0].astype('int').max()

    for n_frame in range(sequence_frames + 1):
        frame_annots = annots[annots[:, 0].astype('int') == n_frame]

        if frame_annots.size > 0:
            objs = []
            for annot in frame_annots:
                box = annot[2:6].astype('int')
                box[2:] += box[:2]
                obj = {
                    "bbox": box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": [poly],
                    "category_id": 0,  # bikes too??
                    "iscrowd": 0
                }
                objs.append(obj)

            frame_name = os.path.join(image_path, "frame_{:04d}.jpg".format(n_frame))
            h, w = cv2.imread(frame_name, 0).shape
            frame = {
                'file_name': frame_name,
                'image_id': n_frame,
                'height': h,
                'width': w,
                'annotations': objs
            }

            dataset.append(frame)

    return dataset


def inference(config_file, dataset, process_images=False, scale_factor=0.25, gt=None):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    predictor = DefaultPredictor(cfg)

    predictions = []

    for image in dataset:
        frame = int(image.split("/")[-1].split(".")[0].split("_")[1])
        im = cv2.imread(image)
        out = predictor(im)['instances']
        boxes = out.get_fields()['pred_boxes']
        preds = out.get_fields()['pred_classes']
        scores = out.get_fields()['scores']

        car_boxes = boxes[preds == 2]
        car_scores = scores[preds == 2]

        # adapt preds to calculate ap method
        # frame, str_category, id (fake)
        prediction_prefix = [frame, 'car', 0]
        preds = torch.cat((car_boxes.tensor, car_scores[..., None]), dim=1)
        preds = preds.to('cpu').numpy().tolist()

        # only add to preds if has detected cars in frame
        if preds:
            preds = [prediction_prefix + pred for pred in preds]
            predictions += preds
        # print(image)

        # if process_images:
        #     h, w = im.shape[:2]
        #     dh, dw = int(h * scale_factor), int(w * scale_factor)
        #     car_boxes.scale(scale_factor, scale_factor)
        #     resized_boxes = car_boxes.tensor.to('cpu').numpy().tolist()
        #     im = cv2.resize(im, (dw, dh))
        #
        #     for b in resized_boxes:
        #         b = [int(a) for a in b]
        #         xy = (b[0], b[1])
        #         x2y2 = (b[2], b[3])
        #         cv2.rectangle(im, xy, x2y2, color=(0, 0, 255))
        #
        #     # cv2.imshow("im", im)
        #     # cv2.waitKey(0)

    return predictions


if __name__ == '__main__':
    # get_AICity_dataset('/home/devsodin/MCV/M6/mcv-m6-2020-team4/datasets/AICity_data/train/S03/c010/data', '/home/devsodin/MCV/M6/mcv-m6-2020-team4/datasets/AICity_data/train/S03/c010/gt/gt.txt')

    from utils.utils import get_files_from_dir


