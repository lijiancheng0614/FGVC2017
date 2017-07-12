import sys
sys.path.insert(0, 'caffe/python')
import caffe
import os
import numpy as np
from PIL import Image

category = sys.argv[1]
assert category in ['dresses', 'outerwear', 'pants', 'shoes']

model_name = sys.argv[2]
model_iteration = sys.argv[3]
gpu_id = int(sys.argv[4])
output_file_path = '{}_submission.csv'.format(category)
image_root = 'data/images/test'
test_list = 'data/list_test.txt'
task_id_label_map_path = 'data/task_id_label_map.txt'

class_names = {'dresses':['age', 'collar', 'color', 'decoration', 'length', 'material', 'occasion', 'pattern', 'silhouette', 'sleeve_length'],
    'outerwear':['age', 'closure_type', 'collar', 'color', 'gender', 'length', 'material', 'pattern', 'sleeve_length', 'type'],
    'pants':['age', 'color', 'decoration', 'fit', 'gender', 'length', 'material', 'pattern', 'type'],
    'shoes':['age', 'back_counter_type', 'closure_type', 'color', 'decoration', 'flat_type', 'gender', 'heel_type', 'material', 'toe_shape', 'type', 'up_height']}
num_task = 45
label_id = {'dresses':[0, 1, 2, 3, 5, 6, 7, 8, 9, 10], # ignore 'gender'
    'outerwear':range(11, 21),
    'pants':[21, 22, 23, 24, 25, 26, 27, 28, 30], # ignore 'rise_type'
    'shoes':[31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44]}# ignore 'boot_type', 'pump_type'

all_label_id = {'dresses':range(0, 11),
    'outerwear':range(11, 21),
    'pants':range(21, 31),
    'shoes':range(31, 45)}

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

def test(class_names, net):
    _ = net.forward()
    out = list()
    for name in class_names:
        out.append(net.blobs['prob_' + name].data[0].argmax())
    return out

fd = open(test_list)
lines = [line.split()[0] for line in fd]
fd.close()
fd = open(task_id_label_map_path)
task_id_label_map = [map(int, line.split()) for line in fd]
fd.close()

prototxt = 'prototxt/{}/{}_deploy.prototxt'.format(model_name, category)
model = 'model/{}/{}_iter_{}.caffemodel'.format(model_name, category, model_iteration)
net = caffe.Net(prototxt, model, caffe.TEST) 

fd = open(output_file_path, 'w')
#fd.write('id,predicted\n')
for line in lines:
    idx = line.split('.')[0]
    print(idx)
    image_path = os.path.join(image_root, line)
    output = [0 for i in range(num_task)]
    for task_idx in all_label_id[category]:
        output[task_id_label_map[task_idx][0] - 1] = task_id_label_map[task_idx][1]

    if os.path.exists(image_path):
        try:
            out = test(class_names[category], net)
            for k, task_idx in enumerate(label_id[category]):
                output[task_id_label_map[task_idx][0] - 1] = task_id_label_map[task_idx][out[k] + 1]
        except Exception as e:
            print('ERROR: {}'.format(e))
            print(image_path)

    for task_idx in all_label_id[category]:
        task_Id = task_id_label_map[task_idx][0]
        fd.write('{}_{},{}\n'.format(idx, task_Id, output[task_Id-1]))

fd.close()
