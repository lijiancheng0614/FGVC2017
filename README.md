# iMaterialist Challenge at FGVC 2017

https://www.kaggle.com/c/imaterialist-challenge-FGVC2017

## Files tree

```
├── data
│   ├── images
│   ├── dresses_list.txt
│   ├── outerwear_list.txt
│   ├── pants_list.txt
│   └── shoes_list.txt
├── model
├── prototxt
├── download_img.py
├── README.md
├── test.sh
└── train.sh
```

## Usage

### Preparation

Install caffe and add `$CAFFE_ROOT/build/tools/` to $PATH.

### Train

Need to train 4 models:

```bash
./train.sh dresses
./train.sh outerwear
./train.sh pants
./train.sh shoe
```

### Test

Run `test.sh`
