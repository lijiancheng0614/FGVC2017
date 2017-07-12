import sys
sys.path.insert(0, 'caffe/python')
import caffe
import os
import numpy as np
from PIL import Image

model_name = sys.argv[1]
model_iteration = sys.argv[2]
gpu_id = int(sys.argv[3])
output_file_path = 'submission.csv'
image_root = 'data/images/test'
test_list = 'data/list_test.txt'
task_id_label_map_path = 'data/task_id_label_map.txt'
categories = ['dresses', 'outerwear', 'pants', 'shoes']
class_names = [['age', 'collar', 'color', 'decoration', 'length', 'material', 'occasion', 'pattern', 'silhouette', 'sleeve_length'],
    ['age', 'closure_type', 'collar', 'color', 'gender', 'length', 'material', 'pattern', 'sleeve_length', 'type'],
    ['age', 'color', 'decoration', 'fit', 'gender', 'length', 'material', 'pattern', 'type'],
    ['age', 'back_counter_type', 'closure_type', 'color', 'decoration', 'flat_type', 'gender', 'heel_type', 'material', 'toe_shape', 'type', 'up_height']]
num_task = 45
label_id = [[0, 1, 2, 3, 5, 6, 7, 8, 9, 10], # ignore 'gender'
    range(11, 21),
    [21, 22, 23, 24, 25, 26, 27, 28, 30], # ignore 'rise_type'
    [31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44]] # ignore 'boot_type', 'pump_type'

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

def test(class_names, net):
    _ = net.forward()
    out = list()
    for name in class_names:
        out.append(net.blobs['prob_' + name].data[0].argmax())
    return out


fd = open(test_list)
lines = [line.strip() for line in fd]
fd.close()
fd = open(task_id_label_map_path)
task_id_label_map = [map(int, line.split()) for line in fd]
fd.close()

prototxts = ['prototxt/{}/{}_deploy.prototxt'.format(model_name, category) for category in categories]
models = ['model/{}/{}_iter_{}.caffemodel'.format(model_name, category, model_iteration) for category in categories]
nets = [caffe.Net(prototxts[i], models[i], caffe.TEST) for i in range(len(categories))] 

fd = open(output_file_path, 'w')
fd.write('id,predicted\n')
for line in lines:
    idx = line.split('.')[0]
    print(idx)
    image_path = os.path.join(image_root, line)
    output = [0 for i in range(num_task)]
    for i in range(num_task):
        output[task_id_label_map[i][0] - 1] = task_id_label_map[i][1]
    if os.path.exists(image_path):
        for i, category in enumerate(categories):
            try:
                out = test(class_names[i], nets[i])
                for k, task_idx in enumerate(label_id[i]):
                    output[task_id_label_map[task_idx][0] - 1] = task_id_label_map[task_idx][out[k] + 1]
            except Exception as e:
                print('ERROR: {}'.format(e))
                print(image_path)
    for i in range(num_task):
        fd.write('{}_{},{}\n'.format(idx, i + 1, output[i]))

fd.close()
