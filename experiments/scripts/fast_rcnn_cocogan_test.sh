#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=40000
    ;;
  pascal_voc2)
    TRAIN_IMDB="voc2_2007_trainval"
    TEST_IMDB="voc2_2007_test"
    PT_DIR="pascal_voc2"
    ITERS=40000
    ;;
  pascal_voc3)
    TRAIN_IMDB="voc3_2007_trainval"
    TEST_IMDB="voc3_2007_test"
    PT_DIR="pascal_voc3"
    ITERS=80000
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=280000
    ;;
  coco2)
    TRAIN_IMDB="coco2_2014_train"
    TEST_IMDB="coco2_2014_minival"
    PT_DIR="coco2"
    ITERS=280000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

set +x
NET_FINAL='/nfs.yoda/xiaolonw/faster_rcnn/xiaolonw/py-faster-rcnn2/output/fast_rcnn_coco/coco2_2014_train/vgg_cnn_m_1024_fast_rcnn2_iter_560000.caffemodel'
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/coco2/VGG_CNN_M_1024/fast_rcnn/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/fast_rcnn_coco_ori.yml \
  ${EXTRA_ARGS}





