"""
@Author: Dandan Shan
@Date: 2019-09-28 23:37:15
@LastEditTime: 2020-02-26 00:44:32
@Description:
"""
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
import argparse
import os
import pdb
import pickle
import pprint
import re
import time
import warnings
from glob import glob
from pathlib import Path
import sys

core_path = os.path.abspath(os.path.join('submodules', 'hand_object_detector'))
if core_path not in sys.path:
    sys.path.insert(0, core_path)

import _init_paths

# patch_rpn.py
from model.rpn.rpn import _RPN

def patched_reshape(x, d):
    x = x.contiguous()  # Add contiguous() before view
    input_shape = x.size()
    x = x.view(
        input_shape[0],
        int(d),
        int(float(input_shape[1] * input_shape[2]) / float(d)),
        input_shape[3]
    )
    return x

# Patch the method
_RPN.reshape = staticmethod(patched_reshape)


from tools.detection_types.detection_types import Detections
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg
from model.utils.config import cfg_from_file
from model.utils.config import cfg_from_list

import cv2
import numpy as np
import torch
from PIL import Image
from imageio import imread
from tqdm import tqdm


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="pascal_voc",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default="submodules/hand_object_detector/cfgs/vgg16.yml",
        type=str,
    )
    parser.add_argument(
        "--net",
        dest="net",
        help="vgg16, res50, res101, res152",
        default="res101",
        type=str,
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--load_dir", dest="load_dir", help="directory to load models", default="submodules/hand_object_detector/models"
    )
    parser.add_argument(
        "--image_dir",
        dest="image_dir",
        help="directory to load images for demo",
        default="images",
    )
    parser.add_argument(
        "--detections_pb",
        help="Path to save detections as protobuf"
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--mGPUs", dest="mGPUs", help="whether use multiple GPUs", action="store_true"
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride")
    parser.add_argument(
        "--parallel_type",
        dest="parallel_type",
        help="which part of model to parallel, 0: all, 1: model before roi pooling",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--checksession",
        dest="checksession",
        help="checksession to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkepoch",
        dest="checkepoch",
        help="checkepoch to load network",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint to load network",
        default=89999,
        type=int,
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )
    parser.add_argument("--thresh_hand", type=float, default=0.1, required=False)
    parser.add_argument("--thresh_obj", default=0.01, type=float, required=False)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        if np.abs(im_scale - 1) < 1e-5:
            im = im_orig
        else:
            im = cv2.resize(
                im_orig,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == "__main__":
    args = parse_args()

    print("Called with args:")
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = torch.cuda.is_available()

    print("Using config:")
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    if args.detections_pb is None:
        pb_path = Path(args.image_dir).name + ".pb2"
    else:
        if Path(args.detections_pb).suffix != '.pb2':
            warnings.warn("Expected --detections_pb to end in .pb2")
        pb_path = args.detections_pb
    print("Saving detections to", pb_path)

    # load model
    if not os.path.exists(args.load_dir):
        raise Exception(
            "There is no input directory for loading network from " + args.load_dir
        )
    load_name = os.path.join(
        args.load_dir,
        "faster_rcnn_{}_{}_{}.pth".format(
            args.checksession, args.checkepoch, args.checkpoint
        ),
    )

    pascal_classes = np.asarray(
        ["__background__", "targetobject", "hand"]
    )  # (3) >>> add obj class here
    args.set_cfgs = [
        "ANCHOR_SCALES",
        "[8, 16, 32, 64]",
        "ANCHOR_RATIOS",
        "[0.5, 1, 2]",
    ]  # (4) >>> add anchor_scales params here

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            pascal_classes, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res101":
        fasterRCNN = resnet(
            pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res50":
        fasterRCNN = resnet(
            pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res152":
        fasterRCNN = resnet(
            pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if torch.cuda.is_available():
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    print("load model successfully!")

    # initilize the tensor holder here.
    num_boxes = torch.zeros([], dtype=torch.long, device=device)
    gt_boxes = torch.zeros([1, 1, 5], device=device)
    box_info = torch.zeros([1, 1, 5], device=device)

    fasterRCNN.to(device)

    thresh_hand = args.thresh_hand
    thresh_obj = args.thresh_obj
    all_detections_pb_strs = []

    fasterRCNN.eval()

    start = time.time()

    print(f"thresh_hand = {thresh_hand}")
    print(f"thnres_obj = {thresh_obj}")

    imglist = sorted(glob(args.image_dir + "/frame_*.jpg"))
    print("Loaded {} images.".format(len(imglist)))

    stride = args.stride

    def get_frame_number(filename):
        return int(re.search(r'frame_(\d+).jpg', filename).groups()[0])

    imglist = [filename for filename in imglist
               if (get_frame_number(filename) - 1) % stride == 0]
    pbar = tqdm(total=len(imglist), unit="image", dynamic_ncols=True)
    for i, im_file in enumerate(imglist):
        frame_number = get_frame_number(im_file)
        total_tic = time.time()
        im_in = cv2.imread(im_file)
        im = im_in
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
        )

        im_data = torch.from_numpy(im_blob).permute(0, 3, 1, 2).to(device)
        im_info = torch.from_numpy(im_info_np).to(device)

        with torch.no_grad():
            gt_boxes.zero_()
            num_boxes.zero_()
            box_info.zero_()

        det_tic = time.time()

        with torch.no_grad():
            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls,
                rpn_loss_box,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                loss_list,
            ) = fasterRCNN(
                im_data, im_info, gt_boxes, num_boxes, box_info
            )  # (8) >>> add bbox_info and loss list
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0]  # hand contact state info
        offset_vector = loss_list[1][
            0
        ]  # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0]  # hand side info (left/right)

        # get hand contact
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                ).to(device) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(
                    device
                )
                if args.class_agnostic:
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        obj_dets, hand_dets = None, None
        for j in range(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if pascal_classes[j] == "hand":
                inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
            elif pascal_classes[j] == "targetobject":
                inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                cls_dets = torch.cat(
                    (
                        cls_boxes,
                        cls_scores.unsqueeze(1),
                        contact_indices[inds],
                        offset_vector.squeeze(0)[inds],
                        lr[inds],
                    ),
                    1,
                )
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if pascal_classes[j] == "targetobject":
                    obj_dets = cls_dets.cpu().numpy()
                if pascal_classes[j] == "hand":
                    hand_dets = cls_dets.cpu().numpy()

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        data_load_time = det_tic - total_tic

        pbar.set_description(
            "{}, Load time: {:.3f}s, Detect time: {:.3f}s, NMS time: {:.3f}s".format(
                os.path.basename(im_file), data_load_time, detect_time, nms_time
            )
        )
        video_id = args.image_dir.split(os.sep)[-2]
        frame_detections = Detections.from_detections(
                video_id,
                frame_number,
                hand_detections=hand_dets,
                object_detections=obj_dets
        )
        all_detections_pb_strs.append(
            frame_detections.to_protobuf().SerializeToString()
        )
        pbar.update(1)
    with open(pb_path, 'wb') as f:
        pickle.dump(all_detections_pb_strs, f)
    pbar.close()
