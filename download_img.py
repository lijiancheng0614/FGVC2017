import os
import json
import time
import subprocess
import multiprocessing
import numpy as np
from PIL import Image
import cv2

def download_image(urls, storage_path):
    if os.path.exists(storage_path):
        try:
            im = Image.open(storage_path)
            im = np.array(im, dtype = np.float32)
            im = cv2.imread(storage_path)
            if im is not None:
                return
            else:
                os.remove(storage_path)
        except Exception as e:
            print('ERROR: {}'.format(e))
            os.remove(storage_path)
    for url in urls:
        try:
            subprocess.call(['wget', url, '-O', storage_path, '-t', '3', '-T', '60'])
            try:
                im = Image.open(storage_path)
                im = np.array(im, dtype = np.float32)
                im = cv2.imread(storage_path)
                if im is not None:
                    return
                else:
                    os.remove(storage_path)
            except Exception as e:
                print('ERROR: {}'.format(e))
                os.remove(storage_path)
        except Exception as e:
            print('ERROR: {}'.format(e))


json_file = 'data/fgvc4_iMat.test.image.json'
output_dir = 'data/images/test/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

fd = open(json_file)
data = json.load(fd)
fd.close()

urls_list = [item['url'] for item in data['images']]
imageId_list = [item['imageId'] for item in data['images']]
tot = len(urls_list)

def worker(start_id):
    for i in range(start_id, tot, cores):
        print('{} {} {}'.format(time.ctime(), start_id, imageId_list[i]))
        download_image(urls_list[i], output_dir + imageId_list[i] + '.jpg')


cores = 32
p = [multiprocessing.Process(target = worker, args = (i,)) for i in range(cores)]
for i in p:
    i.start()

while len(os.listdir(output_dir)) < tot:
    p = subprocess.Popen('ps aux | grep wget | grep -v grep | awk \'{print $2,$9}\'',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.split('\n')
    for line in out:
        if line == '':
            continue
        pid, start_time = line.split()
        try:
            h0, m0 = start_time.split(':')
            h0, m0 = int(h0), int(m0)
            now = time.localtime(time.time())
            h, m = now.tm_hour, now.tm_min
            if h * 60 + m - (h0 * 60 + m0) > 5:
                _ = subprocess.call(['kill', pid])
        except:
            pass
