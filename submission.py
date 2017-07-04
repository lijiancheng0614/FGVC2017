import sys
import os
import json
import numpy as np
from PIL import Image

num_task = 45

output_file_path = 'submission.csv'

test_list = 'data/list_test.txt'
task_id_label_map_path = 'data/task_id_label_map.txt'

categories = ['dresses', 'outerwear', 'pants', 'shoes']

class_names = {'dresses':['age', 'collar', 'color', 'decoration', 'length', 'material', 'occasion', 'pattern', 'silhouette', 'sleeve_length'],
    'outerwear':['age', 'closure_type', 'collar', 'color', 'gender', 'length', 'material', 'pattern', 'sleeve_length', 'type'],
    'pants':['age', 'color', 'decoration', 'fit', 'gender', 'length', 'material', 'pattern', 'type'],
    'shoes':['age', 'back_counter_type', 'closure_type', 'color', 'decoration', 'flat_type', 'gender', 'heel_type', 'material', 'toe_shape', 'type', 'up_height']}
num_task = 45

all_label_id = {'dresses':range(0, 11), # 'dresses' 11ge
    'outerwear':range(11, 21),#'outerwear' 10ge
    'pants':range(21, 31), #'pants' 10ge
    'shoes':range(31, 45)}#'shoes' 14ge 

fd = open(test_list)
lines = [line.strip() for line in fd]
fd.close()
fd = open(task_id_label_map_path)
task_id_label_map = [map(int, line.split()) for line in fd]
fd.close()
#######----------------------###############
sub_data = {}
fd = open('dresses_submission.csv')
sub_data["dresses"] = [line.strip().split(',') for line in fd]
fd.close()
fd = open('outerwear_submission.csv')
sub_data["outerwear"] = [line.strip().split(',') for line in fd]
fd.close()
fd = open('pants_submission.csv')
sub_data["pants"] = [line.strip().split(',') for line in fd]
fd.close()
fd = open('shoes_submission.csv')
sub_data["shoes"] = [line.strip().split(',') for line in fd]
fd.close()
#########-------------------------############
id_label = {}
fd = open('data/fgvc4_iMat.label_map.json')
label_map = json.load(fd)["labelInfo"]
for item in label_map:
    id_label[item["labelId"]] = item["labelName"]
fd.close()

id_task = {}
fd = open('data/fgvc4_iMat.task_map.json')
task_map = json.load(fd)["taskInfo"]
for item in task_map:
    id_task[item["taskId"]] = item["taskName"]
fd.close()

fd = open(output_file_path, 'w')
fd.write('id,predicted\n')
for line in lines:
    idx = line.split('.')[0]
    print(idx)
    output = [0 for i in range(num_task)]
    for i in range(num_task):
        output[task_id_label_map[i][0] - 1] = task_id_label_map[i][1]
    
        for i, category in enumerate(categories):
            data = sub_data[category]
            #out = test(image_path, class_names[i], nets[i])
            for k, task_idx in enumerate(all_label_id[category]):
                output[task_id_label_map[task_idx][0] - 1] = data[(int(idx)-1)*len(all_label_id[category])+k][1]

    for i in range(num_task):
        fd.write('{}_{},{}\n'.format(idx, i + 1, output[i]))
        #taskName = id_task["{}".format(i + 1)]
        #labelName = id_label["{}".format(output[i])]
        #fd.write('{}_{},{},{},{}\n'.format(idx, i + 1, output[i], taskName, labelName))

fd.close()