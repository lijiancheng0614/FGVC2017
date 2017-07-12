import sys
import os
import json
import numpy as np
from PIL import Image

num_task = 45

output_file_path = 'submission.csv'

test_list = 'data/list_test_all.txt'
task_id_label_map_path = 'data/task_id_label_map.txt'

categories = ['dresses', 'outerwear', 'pants', 'shoes']

class_names = {'dresses':['age', 'collar', 'color', 'decoration', 'length', 'material', 'occasion', 'pattern', 'silhouette', 'sleeve_length'],
    'outerwear':['age', 'closure_type', 'collar', 'color', 'gender', 'length', 'material', 'pattern', 'sleeve_length', 'type'],
    'pants':['age', 'color', 'decoration', 'fit', 'gender', 'length', 'material', 'pattern', 'type'],
    'shoes':['age', 'back_counter_type', 'closure_type', 'color', 'decoration', 'flat_type', 'gender', 'heel_type', 'material', 'toe_shape', 'type', 'up_height']}
num_task = 45

all_label_id = {'dresses':range(0, 11),
    'outerwear':range(11, 21),
    'pants':range(21, 31),
    'shoes':range(31, 45)}

fd = open(test_list)
lines = [line.split('.')[0] for line in fd]
fd.close()
fd = open(task_id_label_map_path)
task_id_label_map = [map(int, line.split()) for line in fd]
fd.close()

sub_data = {}
for category in categories:
    fd = open(category + '_submission.csv')
    for line in fd:
        line = line.strip().split(',')
        sub_data[line[0]] = line[1]
    fd.close()

output_default = [0 for i in range(num_task)]
for i in range(num_task):
    output_default[task_id_label_map[i][0] - 1] = task_id_label_map[i][1]
fd = open(output_file_path, 'w')
fd.write('id,predicted\n')
for idx in lines:
    print(idx)
    for i in range(num_task):
        task_id = '{}_{}'.format(idx, i + 1)
        if task_id in sub_data:
            fd.write('{},{}\n'.format(task_id, sub_data[task_id]))
        else:
            fd.write('{},{}\n'.format(task_id, output_default[i]))

fd.close()