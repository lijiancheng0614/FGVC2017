#!/usr/bin/env sh

now=$(date +"%Y%m%d_%H%M%S")
FILE_SOLVER=prototxt/$2/$1_solver.prototxt
DIR_MODEL=model/$2
FILE_INIT_MODEL=$DIR_MODEL/$2.caffemodel
DIR_LOG=log/$2
mkdir -p $DIR_LOG
FILE_LOG=$DIR_LOG/$1_train_$now.txt

caffe train --solver="$FILE_SOLVER" \
    --weights="$FILE_INIT_MODEL" \
    --gpu=$3 \
2>&1 | tee "$FILE_LOG" &
