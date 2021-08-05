import sys
from scipy.spatial.transform import Rotation as R
import argparse
import os
import pandas as pd
import numpy as np


sys.path.insert(1, '../utils/')
from obj_loader import obj_loader
from data_helper import *

def write_line(fout, line):
	fout.write(line + '\n')

def normalize_one_obj(in_obj_dir, out_obj_dir, rpy=None, degrees=True):
	vertex_normalized, _, _ = obj_loader(in_obj_dir, normalize=True)

	if rpy is not None:
		r = R.from_euler('xyz', rpy, degrees=degrees)
		vertex_normalized = r.apply(vertex_normalized)
	
	fout = open(out_obj_dir, 'w+')

	with open(in_obj_dir, 'r') as fin:
		v_i = 0
		for line in fin:
			if line.startswith('#'):
				write_line(fout, line)
				continue
			values = line.split()
			if len(values) < 1:
				write_line(fout, line)
				continue
			if values[0] != 'v':
				write_line(fout, line)
				continue
			
			write_line(fout, 'v ' + ' '.join(['{:.5f}'.format(tmp) for tmp in vertex_normalized[v_i]]))
			v_i += 1
		assert v_i == vertex_normalized.shape[0]
	fout.close()

def normalize_obj(unzip_dir, labels_dir, rotate, use_labels, use_labels_txt, labels_txt_dir=None):
	if use_labels:
		if rotate:
			obj_id_arr, obj_rpy_arr = load_labels_by_keys(labels_dir, ['id', 'rpy'])
		else:
			obj_id_arr =  load_labels_by_keys(labels_dir, ['id'])[0]
	elif use_labels_txt:
		if rotate:
			obj_id_arr, obj_rpy_arr = load_labels_txt_by_keys(labels_txt_dir, ['id', 'rpy'])
		else:
			obj_id_arr =  load_labels_txt_by_keys(labels_txt_dir, ['id'])[0]
	else:
		obj_id_arr = get_int_folders_in_dir(unzip_dir, int_only=True)

	for i, obj_id in enumerate(obj_id_arr):
		obj_folder = os.path.join(unzip_dir, str(obj_id))
		if not os.path.isdir(obj_folder):
			print('{} does not exist! skipping'.format(obj_folder))
			continue

		obj_dir = os.path.join(unzip_dir, obj_folder, 'model_meshlabserver.obj')
		obj_normalized_dir = obj_dir[:-4] + '_normalized.obj'
		# if not os.path.exists(obj_normalized_dir) and os.path.exists(obj_dir):
		if not os.path.exists(obj_normalized_dir) and os.path.exists(obj_dir):
			if rotate:
				obj_rpy = obj_rpy_arr[i]
				normalize_one_obj(obj_dir, obj_normalized_dir, obj_rpy)
				print('normalized & rotated', obj_dir)
			else:
				normalize_one_obj(obj_dir, obj_normalized_dir)
				print('normalized', obj_dir)



if __name__ == "__main__":
	# assert len(sys.argv) == 2
	# in_obj_dir = sys.argv[1]
	# in_obj_dir = '/home/yifany/geo_data/hook_wall/test/model_meshlabserver.obj'
	# out_obj_dir = in_obj_dir[:-4] + '_normalized.obj'
	# out_obj_r_dir = in_obj_dir[:-4] + '_normalized_rotated.obj'
	# normalize_obj(in_obj_dir, out_obj_r_dir, [90, 0, 90])

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="/home/yifany/geo_data/")
	parser.add_argument('--labels_folder_dir', default='/home/yifany/geo_data/labels/')
	parser.add_argument('--category')
	parser.add_argument('--no_rotate', action='store_true')
	parser.add_argument('--use_labels', action='store_true')
	parser.add_argument('--use_labels_txt', action='store_true')
	args = parser.parse_args()

	unzip_dir = os.path.join(args.data_dir, args.category)
	labels_dir = os.path.join(args.data_dir, 'labels', args.category + '.csv')

	if (not args.use_labels) and (not args.use_labels_txt) :
		args.no_rotate = True

	labels_txt_dir = os.path.join(args.labels_folder_dir, '{}_rpy_scaling.txt'.format(args.category))
	print(labels_txt_dir)
	if args.use_labels_txt:
		assert os.path.isfile(labels_txt_dir)
	
	normalize_obj(unzip_dir, labels_dir, (not args.no_rotate), args.use_labels, args.use_labels_txt, labels_txt_dir)







