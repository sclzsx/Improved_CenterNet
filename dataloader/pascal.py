import os
import cv2
import json
import math
import pickle
import numpy as np
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
# import pycocotools.coco as coco

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius

VOC_NAMES = ['__background__', "soil"]
VOC_MEAN = [0.485, 0.456, 0.406]
VOC_STD = [0.229, 0.224, 0.225]

VOC_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
VOC_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                     [-0.5832747, 0.00994535, -0.81221408],
                     [-0.56089297, 0.71832671, 0.41158938]]


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_NAMES, range(len(VOC_NAMES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class PascalVOC(data.Dataset):
    def __init__(self, data_dir, img_size=384, split='train', **kwargs):
        super(PascalVOC, self).__init__()
        self.num_classes = len(VOC_NAMES)
        self.class_names = VOC_NAMES
        self.valid_ids = np.arange(1, self.num_classes, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.data_rng = np.random.RandomState(123)
        self.mean = np.array(VOC_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(VOC_STD, dtype=np.float32).reshape(1, 1, 3)
        self.eig_val = np.array(VOC_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(VOC_EIGEN_VECTORS, dtype=np.float32)

        self.split = split
        self.data_dir = data_dir
        self.annotation_parser = AnnotationTransform()
        self.idx = list()
        self.images = os.path.join(self.data_dir, "JPEGImages/%s.jpg")
        self.annotations = os.path.join(self.data_dir, "Annotations/%s.xml")
        txt = os.path.join(self.data_dir, "ImageSets/Main/%s.txt" % self.split)
        with open(txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            self.idx.append(line)

        self.max_objs = 100
        self.padding = 31  # 127 for hourglass
        self.down_ratio = 4
        if img_size == 240:
            self.img_size = {'h': 256, 'w': 192}
        else:
            self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

    def __getitem__(self, index):
        img_id = self.idx[index]
        img_path = self.images % (img_id)
        annotation_path = self.annotations % (img_id)
        # print(annotation_path)
        target = ET.parse(annotation_path).getroot()
        targets = self.annotation_parser(target)

        labels = targets[:, 4].astype(np.int32)
        # print(labels)
        bboxes = targets[:, :4]
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0

        flipped = False
        if self.split == 'train':
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        img = (img.astype(np.float32) / 255.)
        if self.split == 'train':
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1

        '''
        cv2.imshow("image", img)
        print(hmap.shape)
        tmp = hmap[7]
        print(tmp.shape)
        cv2.imshow("hm",tmp)
        cv2.waitKey(0)
        '''

        return {'image': img, 'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds,
                'ind_masks': ind_masks, 'c': center, 's': scale, 'img_id': img_id}

    def __len__(self):
        return len(self.idx)

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self.images % img_id, cv2.IMREAD_COLOR)


class PascalVOC_eval(PascalVOC):
    def __init__(self, data_dir, split, img_size, test_scales=(1,), test_flip=False, fix_size=True, **kwargs):
        super(PascalVOC_eval, self).__init__(data_dir, img_size, split, **kwargs)
        self.test_flip = test_flip
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, index):
        # img_id = self.images[index]
        # img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        img_id = self.idx[index]
        img_path = self.images % (img_id)
        image = cv2.imread(img_path)
        height, width = image.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = height * scale
            new_width = width * scale

            if self.fix_size:
                img_height, img_width = self.img_size['h'], self.img_size['w']
                center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                scaled_size = max(height, width) * 1.0
                scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | self.padding) + 1
                img_width = (new_width | self.padding) + 1
                center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                scaled_size = np.array([img_width, img_height], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            # cv2.imshow("image1", img)
            trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            # cv2.imshow("image2", img)
            # cv2.waitKey(0)

            img = img.astype(np.float32) / 255.
            img -= self.mean
            img /= self.std
            img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

            if self.test_flip:
                img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

            out[scale] = {'image': img,
                          'center': center,
                          'scale': scaled_size,
                          'fmap_h': img_height // self.down_ratio,
                          'fmap_w': img_width // self.down_ratio}

        return img_id, out

    def convert_eval_format(self, all_bboxes):
        # all_bboxes: num_samples x num_classes x 5
        detections = [[] for _ in self.class_names[1:]]
        for i in range(len(self.idx)):
            # img_id = self.images[i]
            # img_name = self.coco.loadImgs(ids=[img_id])[0]['file_name'].split('.')[0]
            img_name = self.idx[i][1]
            for j in range(1, self.num_classes + 1):
                if len(all_bboxes[img_name][j]) > 0:
                    for bbox in all_bboxes[img_name][j]:
                        detections[j - 1].append((img_name, bbox[-1], *bbox[:-1]))
        print(img_name)
        print(detections[8])
        detections = {cls: det for cls, det in zip(self.class_names[1:], detections)}
        return detections

    def run_eval(self, results, save_dir=None):
        detections = self.convert_eval_format(results)
        if save_dir is not None:
            torch.save(detections, os.path.join(save_dir, 'results.t7'))
        eval_map = eval_mAP(os.path.join(self.data_dir, self.idx[0][0]))
        aps, map = eval_map.do_python_eval(detections)
        return map, aps

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
            if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
        return out


class eval_mAP:
    def __init__(self, VOC_test_root, set='test'):
        self.VOC_root = VOC_test_root
        # self.YEAR = YEAR
        self.set_type = set
        self.annopath = os.path.join(VOC_test_root, 'Annotations', '{:s}.xml')
        self.imgpath = os.path.join(VOC_test_root, 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(VOC_test_root, 'ImageSets', 'Main', '%s.txt')
        self.devkit_path = os.path.join(VOC_test_root)

    def parse_record(self, filename):
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

    def do_python_eval(self, detections, use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')

        aps = []
        # The PASCAL VOC metric changed in 2010
        print('use VOC07 metric ' if use_07 else 'use VOC12 metric ')

        for i, cls in enumerate(VOC_NAMES[1:]):
            rec, prec, ap = self.voc_eval(detections[cls], self.annopath,
                                          self.imgsetpath % self.set_type,
                                          cls, cachedir, ovthresh=0.5, use_07_metric=use_07)
            aps += [ap]
            print('AP for %s = %.2f%%' % (cls, ap * 100))

        print('Mean AP = %.2f%%' % (np.mean(aps) * 100))
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
        return aps, np.mean(aps)

    def voc_ap(self, recall, precision, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([0.], precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def voc_eval(self,
                 cls_detections,
                 annopath,
                 imagesetfile,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=False,
                 use_difficult=False):
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
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile):
            # load annotations
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_record(annopath.format(imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                try:
                    recs = pickle.load(f)
                except:
                    recs = pickle.load(f, encoding='bytes')

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            if use_difficult:
                difficult = np.array([False for x in R]).astype(np.bool)
            else:
                difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        image_ids = [x[0] for x in cls_detections]
        confidence = np.array([float(x[1]) for x in cls_detections])
        BB = np.array([[float(z) for z in x[2:]] for x in cls_detections])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
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
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

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
        ap = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap


if __name__ == '__main__':
  data_dir = '/workspace/DATA/zhatu/zhatu0814/voc'
  dataset = PascalVOC(data_dir)
  for data in dataset:
      pass