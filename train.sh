#!/usr/bin/env sh

now=$(date +"%Y%m%d_%H%M%S")
FILE_SOLVER=prototxt/inception_resnet_v2/$1_train.prototxt
FILE_INIT_MODEL=model/inception_resnet_v2.caffemodel
DIR_LOG=logs/inception_resnet_v2
mkdir -p $DIR_LOG
FILE_LOG=$DIR_LOG/$1_train_$now.txt

caffe train --solver="$FILE_SOLVER" \
    --weights="$FILE_INIT_MODEL" \
2>&1 | tee "$FILE_LOG" &
