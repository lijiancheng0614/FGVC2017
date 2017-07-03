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

- Environment

    Install caffe and add `$CAFFE_ROOT/build/tools/` to $PATH.

    `make pycaffe` and add `$CAFFE_ROOT/python/` to $PYTHONPATH, or `ln -s $CAFFE_ROOT/python caffe/`.

- Dataset

    https://github.com/visipedia/imat_comp

    - Download the dataset files here:

        [iMaterialist dataset [8.0GB]](https://storage.googleapis.com/imat/imat_dataset2017.tar.gz)

        Number of missing images: 1635 / 42029 (3.89%) in training, 298 / 8432 (3.53%) in validation and 1321 / 33726 (3.92%) in test set.

    - Or download datasets by `python download_img.py`.

### Train

Run train script: `./train.sh $CATEGORY $MODEL_NAME $GPU_ID`

```bash
./train.sh dresses inception_v3 0
./train.sh outerwear inception_v3 1
./train.sh pants inception_v3 2
./train.sh shoes inception_v3 3
```

### Test

- Run test script: `python test.py $CATEGORY $MODEL_NAME $MODEL_ITERATION $PHASE $GPU_ID`

    ```bash
    python test.py dresses inception_v3 33000 train 0
    python test.py outerwear inception_v3 42000 val 1
    ...
    ```

- Run predict script for submission

    ```bash
    python predict.py inception_v3 33000 0
    ```