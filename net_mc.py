import os

from tqdm import tqdm
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from torch import optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from utils.data import read_gt_txt, filter_det_confidence, read_detections_file
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
import pytorch_metric_learning.utils.logging_presets as logging_presets


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


def generate_car_crops(root_dir, cameras):
    for camera in cameras:
        dets = np.array(read_gt_txt(os.path.join(root_dir, camera, "gt", "gt.txt")))
        out_dir = os.path.join(root_dir, "crops")
        os.makedirs(out_dir, exist_ok=True)
        get_crops_from_dets(os.path.join(root_dir, camera, "images"), dets, out_dir, camera)


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


def cut_det(im, box):
    cutted = im[box[1]: box[3], box[0]:box[2], :]
    cutted = cv2.resize(cutted, (224, 224))
    base_tfms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cutted = base_tfms(cutted)

    return cutted


def get_camera_embeddings(model, camera_folder, use_dets=True):
    embeddings = []
    if use_dets:
        dets = np.array(filter_det_confidence(read_detections_file(os.path.join(camera_folder, "det", "det_mask_rcnn.txt"))))
        frames = np.unique(dets[:, 0].astype(int))
    else:
        predictor = get_detector()
        frames = len(os.listdir(os.path.join(camera_folder, "images")))

    for image in sorted(os.listdir(os.path.join(camera_folder, "images"))):
        frame = int(image.split(".")[0].split("_")[-1])

        im = cv2.imread(os.path.join(camera_folder, "images", image))
        if use_dets:
            if frame not in frames:
                embeddings.append([])
            else:
                frame_dets = dets[dets[:, 0].astype(int) == frame]
                frame_embeds = []
                for det in frame_dets:
                    box = det[-5:-1].astype(float).astype(int)
                    cutted_im = cut_det(im, box)
                    cutted_im = cutted_im.cuda()

                    frame_embeds.append(model.get_embedding(cutted_im))

                embeddings.append(torch.stack(frame_embeds))
        else:
            preds = perform_det(predictor, im)

            if len(preds) == 0:
                embeddings.append([])
            else:
                frame_embeds = []
                for box in preds:
                    cutted_im = cut_det(im, box)
                    cutted_im = cutted_im.cuda()

                    frame_embeds.append(model.get_embedding(cutted_im))

                embeddings.append(frame_embeds)

    return embeddings




if __name__ == '__main__':

    base_tfms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_tfms = T.Compose([T.RandomHorizontalFlip(), base_tfms])

    train_dataset = ImageFolder("datasets/m6_data/train", transform=train_tfms)
    val_dataset = ImageFolder("datasets/m6_data/test", transform=base_tfms)

    trunk = Net(embedding_size=256)
    trunk = trunk.cuda()


    # Set the loss function
    loss = losses.TripletMarginLoss(margin=0.01)

    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.00005)


# Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=3200)

# Set other training parameters
    batch_size = 64
    num_epochs = 200

# Package the above stuff into dictionaries.
    models = {"trunk": trunk}
    optimizers = {"trunk_optimizer": trunk_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}

    record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": val_dataset}
    model_folder = "example_saved_models"

# Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook)
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
    trainer = trainers.MetricLossOnly(models,
                                    optimizers,
                                    batch_size,
                                    loss_funcs,
                                    mining_funcs,
                                    train_dataset,
                                    sampler=sampler,
                                    end_of_iteration_hook=hooks.end_of_iteration_hook,
                                    end_of_epoch_hook=end_of_epoch_hook)

    trainer.train(num_epochs=num_epochs)


    root_dir = "/home/devsodin/Downloads/AIC20_track3_MTMC/AIC20_track3/train/S03"
    cameras = os.listdir(root_dir)

    camera_embeddings = {}
    model = torch.load('model_256.pt')
    model = model.cuda()
    model.eval()

    for camera in cameras:
        dir = os.path.join(root_dir, camera)
        camera_embeddings[camera] = get_camera_embeddings(model, dir)


