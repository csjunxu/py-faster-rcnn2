# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc2 import pascal_voc2
from datasets.pascal_voc3 import pascal_voc3
from datasets.pascal_voc4 import pascal_voc4
from datasets.nyud2_voc import nyud2_voc
from datasets.nyud3_voc import nyud3_voc
from datasets.coco import coco
from datasets.coco2 import coco2
import numpy as np


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2015']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'nyud2_images_{:s}_{:s}'.format( year, split)
        __sets[name] = (lambda split=split, year=year: nyud2_voc(split, year))

for year in ['2015']:
    for split in [ 'trainval', 'test']:
        name = 'nyud3_images_{:s}_{:s}'.format( year, split)
        __sets[name] = (lambda split=split, year=year: nyud3_voc(split, year))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007']:
    for split in [ 'trainval', 'test']:
        name = 'voc2_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc2(split, year))

for year in ['2007']:
    for split in [ 'trainval', 'test']:
        name = 'voc3_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc3(split, year))

for year in ['2007']:
    for split in [ 'trainval' ]:
        name = 'voc4_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc4(split, year))



# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train']:
        name = 'coco2_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco2(split, year))


# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
