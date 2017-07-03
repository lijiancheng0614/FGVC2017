import json
import os
import urllib
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def download_images(urls, storage_path, index):
	if index % 20 == 0:
		print(index)

	if os.path.exists(storage_path):
		try:
			im = Image.open(storage_path)
			im = np.array(im, dtype = np.float32)
			return 2
		except:
			os.remove(storage_path)

	for url in urls:
		try:
			f = urllib.urlopen(url) 
			with open(storage_path, "wb") as code:
   				code.write(f.read())
   			try:
				im = Image.open(storage_path)
				im = np.array(im, dtype = np.float32)
				return 1
			except:
				os.remove(storage_path)
				continue
		except:
				continue

#fgvc4_iMat.test.image
#fgvc4_iMat.train.data
#fgvc4_iMat.validation.data

with open('fgvc4_iMat.test.image.json', 'r') as f:
	data = json.load(f)
	
	urls_list = [item['url'] for item in data['images']]
	path_names = ["data/test_data/" + item['imageId'] + ".jpg" for item in data['images']]
	indexes = [i for i in range(1, len(path_names)+1)]

	with ThreadPoolExecutor(128) as executor:
		executor.map(download_images, urls_list, path_names, indexes)

print("finish!")