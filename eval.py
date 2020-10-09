"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import os
import sys
import time

sys.path.append(os.getcwd())

import torch
from dataloader.pascal import PascalVOC_eval
from dataloader.pascal import VOC_NAMES as labelmap

import argparse
import numpy as np
import pickle
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from nets.hourglass import get_hourglass

# from nets.resdcn import get_pose_net
# from nets.resnet import get_pose_net
# from nets.resnet_optimG import get_pose_net
# from nets.resnet_optim import get_pose_net
# from nets.ResNet_FPN import get_pose_net
# from nets.MobileNetv2G import get_pose_net
# from nets.CenterFace_MV2 import get_pose_net
# from nets.MobileNetSSH import get_pose_net
# from nets.resnet_optimG import get_pose_net
# from nets.hrnet_vggv2 import get_pose_net
# from nets.vgg_optim import get_pose_net
from nets.Unet import get_pose_net

from utils.utils import load_model
from utils.image import transform_preds
from utils.summary import create_logger
from utils.post_process import ctdet_decode
from nms.nms import soft_nms


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/workspace/DATA/zhatu/zhatu0814/voc')
parser.add_argument('--log_name', type=str, default='centernet_unet_2020-09-02-09')

parser.add_argument('--dataset', type=str, default='pascal', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='resnet_18')

parser.add_argument('--img_size', type=int, default=320)

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--retest', type=bool, default=False)

parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')

parser.add_argument('--over_thresh', default=0.3, type=float,
                    help='Cleanup and remove results files following eval')

cfg = parser.parse_args()

# train_sets = [('buscy', 'test')]

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt')
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'centernet_unet_2020-09-02-09/checkpoint_140.t7')


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(data_dir, image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(data_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(data_dir, all_boxes, dataset, set_type):
    for cls_ind, cls in enumerate(labelmap):
        if cls_ind == 0:
            continue
        # get any class to store the result
        filename = get_voc_results_file_template(data_dir, set_type, cls)
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(dataset.idx):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s}*{:.3f}*{:.1f}*{:.1f}*{:.1f}*{:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir, data_dir, set_type, use_07=False, iou=0.5):
    cachedir = os.path.join(output_dir, 'annotations_cache')
    imgsetpath = os.path.join(data_dir, 'ImageSets', 'Main', 'test.txt')
    annopath = os.path.join(data_dir, 'Annotations', '%s.xml')
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # print(cachedir)
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    for i, cls in enumerate(labelmap):
        if i == 0:
            continue
        # if i is not 7 or i is not 15:
        #    continue
        # print(i, cls)
        filename = get_voc_results_file_template(output_dir, set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=iou, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        # pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    if iou == 0.5:
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
    return np.mean(aps)

    # return aps


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.3,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    # print(imagesetfile)
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # save the truth data as pickle,if the pickle in the file, just load it.
    # if not os.path.isfile(cachefile):
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split('*') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, dataset, model):
    # the len of pic
    num_images = len(dataset)
    # all detections are collected into:[21,4952,0]
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]

    det_file = os.path.join(save_folder, 'detections.pkl')

    if cfg.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        return all_boxes

    max_per_image = 100
    print(num_images)
    for i in tqdm(range(num_images)):
        with torch.no_grad():

            img_id, inputs = dataset[i]

            detections = []
            for scale in inputs:
                inputs[scale]['image'] = torch.from_numpy(inputs[scale]['image']).to(cfg.device)

                output = model(inputs[scale]['image'])[-1]
                dets = ctdet_decode(*output, K=cfg.test_topk)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

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
                for j in range(dataset.num_classes):
                    inds = (cls == j)
                    top_preds[j] = dets[inds, :5].astype(np.float32)
                    top_preds[j][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {}
            for j in range(1, dataset.num_classes):
                bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
                if len(dataset.test_scales) > 1:
                    soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, dataset.num_classes)])

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, dataset.num_classes):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            for j in range(1, len(labelmap)):
                c_dets = bbox_and_scores[j]
                all_boxes[j][i] = c_dets

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    return all_boxes


def evaluate_detections(cache_dir, data_dir, box_list, dataset, eval_type='test'):
    # write the det result to dir
    write_voc_results_file(cache_dir, box_list, dataset, eval_type)
    IouTh = np.linspace(.3, 0.95, np.round((0.95 - .3) / .05) + 1, endpoint=True)
    mAPs = []
    for iou in IouTh:
        print(iou)
        mAP = do_python_eval(cache_dir, data_dir, eval_type, use_07=False, iou=iou)
        mAPs.append(mAP)

    print('--------------------------------------------------------------')
    print('map_5095:', np.mean(mAPs))
    print('map_50:', mAPs[0])
    print('--------------------------------------------------------------')
    return np.mean(mAPs), mAPs[0]


if __name__ == '__main__':
    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    # load data
    VOCroot = '/workspace/DATA/zhatu/zhatu0814/voc'
    dataset = PascalVOC_eval(VOCroot, split='test', img_size=cfg.img_size, test_flip=cfg.test_flip)

    # print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = get_hourglass[cfg.arch]
    elif 'resdcn' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=dataset.num_classes)
    elif 'resnet' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=dataset.num_classes)
    else:
        raise NotImplementedError

    # print(model)

    model = load_model(model, cfg.pretrain_dir)
    model = model.to(cfg.device)
    model.eval()

    # evaluation
    cache_path = cfg.save_folder
    data_path = VOCroot

    all_boxes = test_net(cache_path, dataset, model)

    print('Evaluating detections')
    result = evaluate_detections(cache_path, data_path, all_boxes, dataset, 'test')
