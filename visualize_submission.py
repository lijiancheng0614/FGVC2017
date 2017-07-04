import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

submission_path = 'submission.csv'
task_map_path = 'data/fgvc4_iMat.task_map.json'
label_map_path = 'data/fgvc4_iMat.label_map.json'

image_root = 'data/images/train/'

fd = open(task_map_path)
task_map_json = json.load(fd)['taskInfo']
fd.close()

task_map = dict()
for line in task_map_json:
    task_map[int(line['taskId'])] = line['taskName']

fd = open(label_map_path)
label_map_json = json.load(fd)['labelInfo']
fd.close()

label_map = dict()
for line in label_map_json:
    label_map[int(line['labelId'])] = line['labelName']

fd = open(submission_path)
lines = [line.strip().split(',') for line in fd][1:]
fd.close()

submission = dict()
for line in lines:
    idx, task_id = map(int, line[0].split('_'))
    if idx not in submission:
        submission[idx] = dict()
    submission[idx][task_id] = int(line[1])

def plot(file_path, iterations):
    im = Image.open(file_path)
    im = np.array(im, dtype=np.uint8)

    plt.figure(figsize=(20, 16))
    plt.subplot(121)
    plt.imshow(im)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(np.zeros((640, 300, 3)))
    height = 14
    for i in range(len(labels)):
        plt.text(0, height * i + height / 2, labels[i], family='Times New Roman', size=14, color='#ffffff')
    plt.axis('off')

    # plt.savefig(idx)
    plt.show()


idx_list = submission.keys()
# random.shuffle(idx_list)
count = 5
for idx in idx_list:
    print(idx)
    labels = list()
    for task_id in submission[idx].keys():
        labels.append('{:23} {}'.format(task_map[task_id], label_map[submission[idx][task_id]]))
        print(labels[-1])
    labels = sorted(labels)
    plot(image_root + str(idx) + '.jpg', labels)
    count -= 1
    if count <= 0:
        break
