import os

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode

from data import read_xml_gt, filter_gt

thing_classes = {
    'car': 0,

}


def get_AICity_dataset(image_path, annot_file, is_train=False, mode='first', train_percent=.25):

    assert os.path.exists(annot_file)

    gt = read_xml_gt(annot_file)
    classes_to_keep = ['car']
    gt = filter_gt(gt, classes_to_keep)

    annots = np.array(gt)
    sequence_frames = np.unique(annots[:, 0].astype('int'))



    dataset = []
    # f = open(annot_file, 'r')
    #
    # annots = [l.split(",") for l in f.readlines()]
    #
    # annots = np.array(annots)
    #
    # sequence_frames = np.unique(annots[:, 0].astype('int'))
    max_train_frames = int(sequence_frames.shape[0] * train_percent)

    if mode == 'first':
        sequence_frames_train = annots[annots[:, 0].astype(int) <= max_train_frames]
        sequence_frames_val = annots[annots[:, 0].astype(int) > max_train_frames]

    # random frames
    # TODO debug for possible bugs
    else:
        np.random.seed(42)
        train_frames = np.random.choice(sequence_frames, max_train_frames, replace=False)
        mask = np.isin(annots[:, 0].astype(int), train_frames)

        sequence_frames_train = sequence_frames[mask, :]
        sequence_frames_val = sequence_frames[~mask, :]

    sequence_frames = sequence_frames_train if is_train else sequence_frames_val
    sequence_frames = sequence_frames[:, 0].astype('int').max()

    for n_frame in range(sequence_frames + 1):
        frame_annots = annots[annots[:, 0].astype('int') == n_frame]

        if frame_annots.size > 0:
            objs = []
            for annot in frame_annots:
                box = annot[3:7].astype('float')
                obj = {
                    "bbox": box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)

            frame_name = os.path.join(image_path, "frame_{:04d}.jpg".format(n_frame + 1))
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


def train(config_file, image_path, annot_file, out_filename="results.txt"):
    train_split = lambda: get_AICity_dataset(image_path, annot_file, mode='first', is_train=True)
    test_split = lambda: get_AICity_dataset(image_path, annot_file, mode='first')

    DatasetCatalog.clear()
    DatasetCatalog.register("ai_city_train", train_split)
    MetadataCatalog.get('ai_city_train').set(thing_classes=[k for k in thing_classes.keys()])

    DatasetCatalog.register("ai_city_test", test_split)
    MetadataCatalog.get('ai_city_test').set(thing_classes=[k for k in thing_classes.keys()])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("ai_city_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 50
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("ai_city_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "ai_city_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    ims_from_test = [annot['file_name'] for annot in test_split()]
    det_bb = inference(config_file, ims_from_test, weight="./output/model_final.pth", save_results=True, out_filename=out_filename)

    return det_bb


def remap_preds(preds):
    remapped = []

    for pred in preds:
        frame = str(pred[0])
        x = pred[3]
        y = pred[4]
        w = str(pred[5] - x)
        h = str(pred[6] - y)
        confidence = str(pred[7])
        remap = [frame, str(-1), str(x), str(y), w, h, confidence, str(-1), str(-1), str(-1)]
        remap = ",".join(remap) + "\n"
        remap
        remapped.append(remap)

    return remapped


def inference(config_file, test_images_filenames, weight=None, save_results=False, out_filename="results.txt"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    if weight is not None:
        cfg.MODEL.WEIGHTS = weight
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    predictor = DefaultPredictor(cfg)

    predictions = []
    # TODO FORMAT frame, -1, xtl, ytl, w, h, confidence, -1,-1,-1
    save_predictions = []
    for image in test_images_filenames:
        frame = int(image.split("/")[-1].split(".")[0].split("_")[1])
        im = cv2.imread(image)
        out = predictor(im)['instances']
        boxes = out.get_fields()['pred_boxes']
        preds = out.get_fields()['pred_classes']
        scores = out.get_fields()['scores']

        # check if doing inference on coco dataset or fine-tuned one
        # TODO change to a better way ??
        car_boxes = boxes[preds == 2] if weight is None else boxes[preds == 0]
        car_scores = scores[preds == 2] if weight is None else scores[preds == 0]

        # adapt preds to calculate ap method
        # frame, str_category, id (fake)
        prediction_prefix = [frame, 'car', 0]
        preds = torch.cat((car_boxes.tensor, car_scores[..., None]), dim=1)
        preds = preds.to('cpu').numpy().tolist()

        # only add to preds if has detected cars in frame
        if preds:
            preds = [prediction_prefix + pred for pred in preds]
            predictions += preds
            if save_results:
                save_predictions += remap_preds(preds)
    if save_results:
        with open(out_filename, 'w') as f:
            f.writelines(save_predictions)

    return predictions


if __name__ == '__main__':
    pass
