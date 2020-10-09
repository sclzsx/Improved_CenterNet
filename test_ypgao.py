import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.utils.data

# from dataloader.coco import COCO_eval
from dataloader.pascal import PascalVOC_eval

from nets.hourglass import get_hourglass
# from nets.resdcn import get_pose_net
# from nets.resnet import get_pose_net
# from nets.resnet_optim import get_pose_net
# from nets.ResNet_FPN import get_pose_net
# from nets.MobileNetv2G import get_pose_net
# from nets.resnet_optimG import get_pose_net
from nets.CenterFace_MV2 import get_pose_net

from utils.utils import load_model
from utils.image import transform_preds
from utils.summary import create_logger
from utils.post_process import ctdet_decode

# from nms.nms import soft_nms
from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from dataloader.pascal import VOC_NAMES as labelmap

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/workspace/DATA/zhatu/zhatu0814/voc/')
parser.add_argument('--log_name', type=str, default='hrnet_vggv2_2020-09-02-09')

parser.add_argument('--dataset', type=str, default='pascal', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='resnet_18')

parser.add_argument('--img_size', type=int, default=320)

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=1)

cfg = parser.parse_args()

source_videos_path = r'/workspace/DATA/zhatu/videos/0829/'
# source_videos_path = r"/opt/cv_storage3/F_public/public/workspace/exchange/zebra_crossing/sampleclips/8mm/shenzhen_bus/20200312/right"
dst_videos_path = r"/workspace/"

VOC_MEAN = [0.485, 0.456, 0.406]
VOC_STD = [0.229, 0.224, 0.225]
os.chdir(cfg.root_dir)
# num_classes = len(labelmap) - 1
num_classes = len(labelmap)
test_scales = (1,)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt')
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'hrnet_vggv2_2020-09-02-09/checkpoint_140.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]


def process_image(image):
    height, width = image.shape[0:2]
    test_scales = (1,)
    fix_size = True
    img_size = cfg.img_size
    img_sizes = {'h': img_size, 'w': img_size}
    down_ratio = 4

    mean = np.array(VOC_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(VOC_STD, dtype=np.float32).reshape(1, 1, 3)

    out = {}
    for scale in test_scales:
        new_height = height * scale
        new_width = width * scale

        if fix_size:
            img_height, img_width = img_sizes['h'], img_sizes['w']
            # print(img_height)
            # print(img_width)
            # print(new_height)
            # print(new_width)
            center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            scaled_size = max(height, width) * 1.0
            # print(scaled_size)
            scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)

        img = cv2.resize(image, (new_width, new_height))
        trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
        img = cv2.warpAffine(img, trans_img, (img_width, img_height))
        # sys.exit()
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

        out[scale] = {'image': img,
                      'center': center,
                      'scale': scaled_size,
                      'fmap_h': img_height // down_ratio,
                      'fmap_w': img_width // down_ratio}

    return out


def detect_image(frame, model):
    inputs = process_image(frame)
    max_per_image = 100
    detections = []
    for scale in inputs:
        inputs[scale]['image'] = torch.from_numpy(inputs[scale]['image']).to(cfg.device)

        output = model(inputs[scale]['image'])[-1]
        dets = ctdet_decode(*output, K=cfg.test_topk)

        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
        print(dets)

        top_preds = {}
        dets[:, :2] = transform_preds(dets[:, 0:2],
                                      inputs[scale]['center'],
                                      inputs[scale]['scale'],
                                      (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                       inputs[scale]['center'],
                                       inputs[scale]['scale'],
                                       (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
        cls = dets[:, -1]
        score = dets[:, -2]

        # print(dets[:, 2:4])
        sys.exit()

        for j in range(num_classes):
            inds = (cls == j)
            top_preds[j] = dets[inds, :5].astype(np.float32)
            top_preds[j][:, :4] /= scale

        detections.append(top_preds)

    bbox_and_scores = {}
    for j in range(1, num_classes):
        bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
        # if len(test_scales) > 1:
        #    soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
    scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, num_classes)])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes):
            keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
            bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

    return bbox_and_scores


def detect_videos(source_videos_path, dst_videos_path, model):
    video_names = os.listdir(source_videos_path)
    for name in video_names:
        videopath = os.path.join(source_videos_path, name)
        dstpath = os.path.join(dst_videos_path, name)

        cap = cv2.VideoCapture(videopath)
        fps = 24  #
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        # resVideoPath = "./" + filepath + ".avi"
        videoWriter = cv2.VideoWriter(dstpath, fourcc, fps, (1920, 1080))
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(dstpath, fourcc, 25.0, (1280, 720))

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                bbox_and_scores = detect_image(frame, model)

                for j in range(1, num_classes):
                    if bbox_and_scores[j].shape[0] > 0:
                        # print(i)
                        box_scores = bbox_and_scores[j][:, 4]
                        box_locations = bbox_and_scores[j][:, 0:4]
                        for i, box_location in enumerate(box_locations):
                            # print(box_location)
                            if box_scores[i] > 0.2:
                                p1 = (int(box_location[0]), int(box_location[1]))
                                p2 = (int(box_location[2]), int(box_location[3]))
                                cv2.rectangle(frame, p1, p2, (0, 255, 0))

                                title = "%s:%.2f" % (labelmap[j], box_scores[i])
                                p3 = (max((p1[0] + p2[0]) // 2, 15), max((p1[1] + p2[1]) // 2, 15))
                                cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

                videoWriter.write(frame)
            else:
                break

        videoWriter.release()


def main():
    logger = create_logger(save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    max_per_image = 100

    print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = get_hourglass[cfg.arch]
    elif 'resdcn' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=num_classes)
    elif 'resnet' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=num_classes)
    else:
        raise NotImplementedError

    model = load_model(model, cfg.pretrain_dir)
    model = model.to(cfg.device)
    model.eval()

    detect_videos(source_videos_path, dst_videos_path, model)


if __name__ == '__main__':
    main()
