# iMaterialist Challenge at FGVC 2017

https://www.kaggle.com/c/imaterialist-challenge-FGVC2017

## Files tree

```
├── data
│   ├── images
│   │   ├── train
│   │   ├── test
│   │   └── val
│   ├── README.md
│   ├── dresses_list_train.txt
│   └── ...
├── log
│   └── inception_v3
│       └── dresses_train_*.txt
├── model
│   ├── inception_v3
│   │   ├── inception_v3.caffemodel
│   │   └── dresses_iter_*.caffemodel
│   ├── inception_resnet_v2
│   │   └── ...
│   └── README.md
├── prototxt
│   ├── inception_v3
│   │   ├── dresses_solver.prototxt
│   │   ├── dresses_train.prototxt
│   │   ├── dresses_deploy.prototxt
│   │   └── ...
│   └── inception_resnet_v2
│       └── ...
├── download_img.py
├── README.md
├── test.py
└── train.sh
```

## Usage

### Preparation

Install caffe and add `$CAFFE_ROOT/build/tools/` to $PATH.

### Train

Run train script: `./train.sh $CATEGORY $MODEL_NAME $GPU_ID`

```bash
./train.sh dresses inception_v3 0
./train.sh outerwear inception_v3 1
./train.sh pants inception_v3 2
./train.sh shoes inception_v3 3
```

### Test

Modifiy `test.py` and run.
