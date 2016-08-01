# ---------------------------------------------------------
# Copyright (c) 2015, Saurabh Gupta
# 
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------

import python_utils.evaluate_detection as eval
import python_utils.general_utils as g_utils
import datasets
import datasets.coco2
import os
import os.path as osp
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from fast_rcnn.config import cfg
from IPython.core.debugger import Tracer



def _filter_crowd_proposals(roidb, crowd_thresh):
    """
    Finds proposals that are inside crowd regions and marks them with
    overlap = -1 (for all gt rois), which means they will be excluded from
    training.
    """
    for ix, entry in enumerate(roidb):
        overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(overlaps.max(axis=1) == -1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        iscrowd = [int(True) for _ in xrange(len(crowd_inds))]
        crowd_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        overlaps[non_gt_inds[bad_inds], :] = -1
        roidb[ix]['gt_overlaps'] = scipy.sparse.csr_matrix(overlaps)
    return roidb

class coco2(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'coco2_' + year + '_' + image_set)
        self._year = year
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path

        clasnum = 80 * 2
        classes = ['__background__']
        for i in range(clasnum):
          classes.append(str(i + 1))
        self._classes=tuple(classes)


        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_type = 'images';
        self._image_set = image_set;
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('mcg')
        self._salt = str(uuid.uuid4())

        # PASCAL specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'crowd_thresh' : 0.7,
                       'min_size' : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = []
        # image_type_list = self._image_type.split('+')
        # for typ in image_type_list:
        #     image_path.append(os.path.join(self._data_path, typ, index + self._image_ext))
        #     assert os.path.exists(image_path[-1]), 'Path does not exist: {}'.format(image_path)

        image_path = os.path.join(self._data_path, self._image_type, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)

        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._devkit_path, 'Annotations_mats',
                                      'splits.mat')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        raw_data = sio.loadmat(image_set_file)[self._image_set].ravel()
        image_index = [now_str[0] for now_str in raw_data]
        
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """

        # return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
        return cfg.DATA_DIR

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        return self._roidb_from_proposals('selective_search')

    def edge_boxes_roidb(self):
        return self._roidb_from_proposals('edge_boxes_AR')

    def mcg_roidb(self):
        return self._roidb_from_proposals('MCG')

    def _roidb_from_proposals(self, method):
        """
        Creates a roidb from pre-computed proposals of a particular methods.
        """
        top_k = self.config['top_k']
        cache_file = osp.join(self.cache_path, self.name +
                              '_{:s}_top{:d}'.format(method, top_k) +
                              '_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{:s} {:s} roidb loaded from {:s}'.format(self.name, method,
                                                            cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        method_roidb = self._load_proposals(method, gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, method_roidb)
        # Make sure we don't use proposals that are contained in crowds
        roidb = _filter_crowd_proposals(roidb, self.config['crowd_thresh'])
    
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote {:s} roidb to {:s}'.format(method, cache_file)
        return roidb

    def _load_proposals(self, method, gt_roidb):
        """
        Load pre-computed proposals in the format provided by Jan Hosang:
        http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
          computing/research/object-recognition-and-scene-understanding/how-
          good-are-detection-proposals-really/
        For MCG, use boxes from http://www.eecs.berkeley.edu/Research/Projects/
          CS/vision/grouping/mcg/ and convert the file layout using
        lib/datasets/tools/mcg_munge.py.
        """
        box_list = []
        top_k = self.config['top_k']
        valid_methods = [
            'MCG',
            'selective_search',
            'edge_boxes_AR',
            'edge_boxes_70']
        assert method in valid_methods

        print 'Loading {} boxes'.format(method)
        for i, index in enumerate(self._image_index):
            if i % 1000 == 0:
                print '{:d} / {:d}'.format(i + 1, len(self._image_index))

            box_file = osp.join(
                cfg.DATA_DIR, 'coco_proposals', method, 'mat',
                self._image_index[i] + '.mat') 

            raw_data = sio.loadmat(box_file)['boxes']
            boxes = np.maximum(raw_data - 1, 0).astype(np.uint16)
            if method == 'MCG':
                # Boxes from the MCG website are in (y1, x1, y2, x2) order
                boxes = boxes[:, (1, 0, 3, 2)]
            # Remove duplicate boxes and very small boxes and then take top k
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            boxes = boxes[:top_k, :]
            box_list.append(boxes)
            # Sanity check
            # im_ann = self._COCO.loadImgs(index)[0]
            # width = im_ann['width']
            # height = im_ann['height']
            # ds_utils.validate_boxes(boxes, width=width, height=height)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._devkit_path, 'Annotations_mats', index + '.mat')
        # print 'Loading: {}'.format(filename)
        raw_data = sio.loadmat(filename)
        objs = raw_data['rec']['objects'][0][0][0]

        # Select object we care about
        objs = [obj for obj in objs if self._class_to_ind.get(str(obj['class'][0])) is not None]
        
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            cls = self._class_to_ind.get(str(obj['class'][0]))
            boxes[ix, :] = obj['bbox'][0] - 1
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (boxes[ix, 2] - boxes[ix, 0] + 1) * (boxes[ix, 3] - boxes[ix, 1] + 1)

        assert (boxes[:, 2] >= boxes[:, 0]).all()

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
    
    def evaluate_detections(self, all_boxes, output_dir, det_salt = '', eval_salt = '', overlap_thresh = 0.5):
      num_classes = self.num_classes
      num_images = self.num_images
      gt_roidb = self.gt_roidb()
      ap = [[]]; prec = [[]]; rec = [[]]
      ap_file = os.path.join(output_dir, 'eval' + det_salt + eval_salt + '.txt')
      with open(ap_file, 'wt') as f:
          for i in xrange(1, self.num_classes):
              dt = []; gt = [];
              # Prepare the output
              for j in xrange(0,num_images):
                  bs = all_boxes[i][j]
                  if len(bs) == 0:
                    bb = np.zeros((0,4)).astype(np.float32)
                    sc = np.zeros((0,1)).astype(np.float32)
                  else:
                    bb = bs[:,:4].reshape(bs.shape[0],4)
                    sc = bs[:,4].reshape(bs.shape[0],1)
                  dtI = dict({'sc': sc, 'boxInfo': bb})
                  dt.append(dtI)
          
              # Prepare the annotations
              for j in xrange(0,num_images):
                  cls_ind = np.where(gt_roidb[j]['gt_classes'] == i)[0]
                  bb = gt_roidb[j]['boxes'][cls_ind,:]
                  diff = np.zeros((len(cls_ind),1)).astype(np.bool)
                  gt.append(dict({'diff': diff, 'boxInfo': bb}))
              bOpts = dict({'minoverlap': overlap_thresh})
              ap_i, rec_i, prec_i = eval.inst_bench(dt, gt, bOpts)[:3]
              ap.append(ap_i[0]); prec.append(prec_i); rec.append(rec_i)
              ap_str = '{:20s}:{:10f}'.format(self.classes[i], ap_i[0]*100)
              f.write(ap_str + '\n')
              print ap_str
          ap_str = '{:20s}:{:10f}'.format('mean', np.mean(ap[1:])*100)
          f.write(ap_str + '\n')
          print ap_str

      eval_file = os.path.join(output_dir, 'eval' + det_salt + eval_salt + '.pkl')
      g_utils.save_variables(eval_file, [ap, prec, rec, self._classes, self._class_to_ind], \
          ['ap', 'prec', 'rec', 'classes', 'class_to_ind'], overwrite = True)
      eval_file = os.path.join(output_dir, 'eval' + det_salt + eval_salt + '.mat')
      g_utils.scio.savemat(eval_file, {'ap': ap, 'prec': prec, 'rec': rec, 'classes': self._classes}, do_compression = True);
      
      return ap, prec, rec, self._classes, self._class_to_ind
 

if __name__ == '__main__':
    d = datasets.coco2('trainval', '2007')
    # res = d.roidb
    from IPython import embed; embed()
