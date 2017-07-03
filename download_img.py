import os
import json
import time
import urllib2
import multiprocessing
import cv2

def download_image(urls, storage_path):
    if os.path.exists(storage_path):
        try:
            im = cv2.imread(storage_path)
            if im is not None:
                return
        except:
            os.remove(storage_path)
    for url in urls:
        try:
            fd = urllib2.urlopen(url)
            open(storage_path, 'wb').write(fd.read())
            try:
                im = cv2.imread(storage_path)
                if im is not None:
                    return
            except:
                os.remove(storage_path)
        except:
            continue

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

cores = 8
p = [multiprocessing.Process(target = worker, args = (i,)) for i in range(cores)]
for i in p:
    i.start()

while len(os.listdir(output_dir)) < tot:
    pass
