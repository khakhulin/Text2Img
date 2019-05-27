import os
import numpy as np
from PIL import Image
import argparse


def find_one_channel_img(file_names, data_path):
	idx = 0
	count = 0
	remove_list = []
	print ("removing 1 channel images ...")
	for file in file_names:
		img = Image.open(os.path.join(data_path, 'images', file.split()[1]))
		if (len((np.array(img)).shape) != 3):
			img.close()
			remove_list.append(file_names[idx].split()[1])
			print ("One channel image: ", file_names[idx])
			del file_names[idx]

		idx += 1

	return remove_list

def remove_one_channel_img(remove_list, data_path):
	remove_set = set(remove_list)

	with open(os.path.join(data_path, 'images.txt'), "r") as f:
		lines = f.readlines()
	
	with open(os.path.join(data_path, 'images.txt'), "w+") as imf:
		for line in lines:
			if ((line.strip("\n").split()[1] in remove_set) == False):
				imf.write(line)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Data path')
	parser.add_argument('--data_path', default='', type=str)
	args = parser.parse_args()


	with open(os.path.join(args.data_path, 'images.txt'), "r") as imf:
		file_names = imf.read().splitlines()

	remove_list = find_one_channel_img(file_names, args.data_path)
	remove_one_channel_img(remove_list, args.data_path)
