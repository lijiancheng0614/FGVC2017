import sys
sys.path.insert(0, 'caffe/python')
import caffe
import os
import numpy as np
from PIL import Image

category = 'dresses'
model_name = 'inception_v3'
gpu_id = 0
image_root = 'data/images/val'
prototxt = 'prototxt/{}/{}_deploy.prototxt'.format(model_name, category)
model = 'model/{}/{}_iter_3000.caffemodel'.format(model_name, category)
test_list = 'data/{}_list_val.txt'.format(category)
if category == 'dresses':
    class_names = ['age', 'collar', 'color', 'decoration', 'length', 'material', 'occasion', 'pattern', 'silhouette', 'sleeve_length']
    # ignore 'gender'
elif category == 'outerwear':
    class_names = ['age', 'closure_type', 'collar', 'color', 'gender', 'length', 'material', 'pattern', 'sleeve_length', 'type']
elif category == 'pants':
    class_names = ['age', 'color', 'decoration', 'fit', 'gender', 'length', 'material', 'pattern', 'type']
    # ignore 'rise_type'
elif category == 'shoe':
    class_names = ['age', 'back_counter_type', 'closure_type', 'color', 'decoration', 'flat_type', 'gender', 'heel_type', 'material', 'toe_shape', 'type', 'up_height']
    # ignore 'boot_type', 'pump_type'

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

def test(image_path, net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    image = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    _ = net.forward()
    out = list()
    for name in class_names:
        out.append(net.blobs['prob_' + name].data[0].argmax())
    return out


net = caffe.Net(prototxt, model, caffe.TEST)

fd = open(test_list, 'r')
lines = fd.readlines()
fd.close()
tot = {name : dict() for name in class_names}
correct = {name : dict() for name in class_names}
for line in lines[1:]:
    line = line.split()
    idx = line[0]
    # print(idx)
    ground_truth = [int(i) for i in line[1:]]
    image_path = os.path.join(image_root, idx)
    out = test(image_path, net)
    for idx, name in enumerate(class_names):
        label = ground_truth[idx]
        if label == -1:
            continue
        if label not in tot[name]:
            tot[name][label] = 0
            correct[name][label] = 0
        tot[name][label] += 1
        if label == out[idx]:
            correct[name][label] += 1

for name in class_names:
    n_correct = sum(correct[name].values())
    n_tot = sum(tot[name].values())
    print('{} {}/{} = {}'.format(name, n_correct, n_tot, n_correct * 1.0 / n_tot if n_tot != 0 else 'NAN'))
    keys = tot[name].keys()
    print(' '.join(['{}/{}'.format(correct[name][k], tot[name][k]) for k in keys]))
    print('')

n_correct = sum([sum(correct[name].values()) for name in class_names])
n_tot = sum([sum(tot[name].values()) for name in class_names])
print('{}/{} = {}'.format(n_correct, n_tot, n_correct * 1.0 / n_tot))
