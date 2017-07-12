import sys
sys.path.insert(0, 'caffe/python')
import caffe
import os
import numpy as np
from PIL import Image

category = sys.argv[1]
model_name = sys.argv[2]
model_iteration_end = int(sys.argv[3])
phase = sys.argv[4]
gpu_id = int(sys.argv[5])
image_root = 'data/images/{}'.format(phase)
test_list = 'data/{}_list_{}.txt'.format(category, phase)
step_size = 3000
if category == 'dresses':
    class_names = ['age', 'collar', 'color', 'decoration', 'length', 'material', 'occasion', 'pattern', 'silhouette', 'sleeve_length']
    gt_map = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    # ignore 'gender'
elif category == 'outerwear':
    class_names = ['age', 'closure_type', 'collar', 'color', 'gender', 'length', 'material', 'pattern', 'sleeve_length', 'type']
    gt_map = range(10)
elif category == 'pants':
    class_names = ['age', 'color', 'decoration', 'fit', 'gender', 'length', 'material', 'pattern', 'type']
    gt_map = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    # ignore 'rise_type'
elif category == 'shoes':
    class_names = ['age', 'back_counter_type', 'closure_type', 'color', 'decoration', 'flat_type', 'gender', 'heel_type', 'material', 'toe_shape', 'type', 'up_height']
    gt_map = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
    # ignore 'boot_type', 'pump_type'

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

def test(net):
    _ = net.forward()
    out = list()
    for name in class_names:
        out.append(net.blobs['prob_' + name].data[0].argmax())
    return out

fd = open('test_iter_{}.txt'.format(category), 'w')
sys.stdout = fd
for model_iteration in range(3000, model_iteration_end + 1, step_size):
    prototxt = 'prototxt/{}/{}_deploy.prototxt'.format(model_name, category)
    model = 'model/{}/old/{}_iter_{}.caffemodel'.format(model_name, category, str(model_iteration))

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
        try:
            out = test(net)
            for idx, name in enumerate(class_names):
                label = ground_truth[gt_map[idx]]
                if label == -1:
                    continue
                if label not in tot[name]:
                    tot[name][label] = 0
                    correct[name][label] = 0
                tot[name][label] += 1
                if label == out[idx]:
                    correct[name][label] += 1
        except Exception as e:
            print('ERROR: {}'.format(e))
            print(image_path)

    print('{} {} iter_{} {}'.format(category, model_name, model_iteration, test_list))
    print('=' * 10)
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

fd.close()
