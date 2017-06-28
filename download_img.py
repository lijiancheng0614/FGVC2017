import json
import os
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor

def download_images(urls, storage_path, index):
	if index % 1000 == 0:
		print("Tried downloading {} images, {} downloaded".format(index, storage_dir = storage_path))

	if os.path.exists(storage_path):
		return 2

	for url in urls:
		try:
			urlretrieve(url, storage_path)
			return 1
		except:
			continue


with open('fgvc4_iMat.test.image.json', 'r') as f:
	data = json.load(f)
	
	urls_list = [item['url'] for item in data['images']]
	path_names = ["test_data/" + item['imageId'] + ".jpg" for item in data['images']]
	indexes = [i for i in range(1, len(path_names)+1)]

	#download_images(urls_list[0], path_names[0], indexes[0])
	#map(download_images, urls_list, path_names, indexes)
	with ThreadPoolExecutor(128) as executor:
		executor.map(download_images, urls_list, path_names, indexes)
