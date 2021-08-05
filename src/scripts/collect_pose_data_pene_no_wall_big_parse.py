import pybullet 
import time
import numpy as np
import random
# np.random.seed(5)
# random.seed(5)
import sys
import os
import argparse
import cv2
import csv
from scipy.spatial.transform import Rotation

from collect_pose_data import PoseDataCollector
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import bullet_client as bc

sys.path.insert(1, '../lin/')
from ompl_lib import calc_penetration
from classifier_dataset_torch import ClassifierDataset

def dict_to_csv(out_dir, all_data):
	print('writing', out_dir)
	w = csv.writer(open(out_dir, "w+"))
	for key, val in all_data.items():
		w.writerow([key, val])
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--sherlock", action='store_true')
	parser.add_argument("--obj_cat_split_id", type=int, default=-1)
	args = parser.parse_args()

	obj_cat_split_id = int(args.obj_cat_split_id)
	if args.sherlock:
		args.home_dir_data = '/scratch/groups/bohg/hang'
		assert args.hook_name != ''
		assert obj_cat_split_id >= 0
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	pos_collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_pene_big_pos')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	neg_collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_pene_big_neg')
	pos_labels_dir = os.path.join(pos_collection_result_folder_dir, 'labels')
	neg_labels_dir = os.path.join(neg_collection_result_folder_dir, 'labels')

	mkdir_if_not(pos_collection_result_folder_dir)
	mkdir_if_not(neg_collection_result_folder_dir)
	mkdir_if_not(pos_labels_dir)
	mkdir_if_not(neg_labels_dir)

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, labels_folder_dir, True, True, with_wall=False)
	p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)

	cp_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'train_list.txt') 
	test_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'test_list.txt')
	train_set = ClassifierDataset(args.home_dir_data, train_list_dir, False, split='train', with_wall=False, one_per_pair=True)
	test_set = ClassifierDataset(args.home_dir_data, test_list_dir, False, split='test', with_wall=False, one_per_pair=True)
	

	if not os.path.exists(neg_collection_result_folder_dir):
		os.mkdir(neg_collection_result_folder_dir)
	# collector = PeneDataCollector(p_id)
	ct = 0
	

	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		
		out_pos_labels_dir = os.path.join(pos_labels_dir, '{}.txt'.format(hook_name))
		out_neg_labels_dir = os.path.join(neg_labels_dir, '{}.txt'.format(hook_name))
		if os.path.exists(out_pos_labels_dir) or os.path.exists(out_neg_labels_dir):
			print('skip', hook_name)
			continue

		num_pos_dict = {}
		num_neg_dict = {}

		for j, object_name in enumerate(all_object_name):
			# if not 'daily_object' in object_name:
				# continue
			result_file_name = hook_name + '_' + object_name
			if (not result_file_name in train_set.all_result_file_names) \
				and (not result_file_name in test_set.all_result_file_names):
				continue

			neg_out_dir = os.path.join(neg_collection_result_folder_dir, result_file_name + '.txt')
			pos_out_dir = os.path.join(pos_collection_result_folder_dir, result_file_name + '.txt')

			if os.path.exists(neg_out_dir):
				neg_result = load_result_file(neg_out_dir)
				num_neg_dict[result_file_name] =  neg_result.shape[0]

			if os.path.exists(pos_out_dir):
				pos_result = load_result_file(pos_out_dir)
				num_pos_dict[result_file_name] =  len(pos_result)
			print(result_file_name)

		out_pos_labels_dir = os.path.join(pos_labels_dir, '{}.txt'.format(hook_name))
		out_neg_labels_dir = os.path.join(neg_labels_dir, '{}.txt'.format(hook_name))
		# collector.p.removeBody(hook_bullet_id)	
		dict_to_csv(out_pos_labels_dir, num_pos_dict)
		dict_to_csv(out_neg_labels_dir, num_neg_dict)


	# for j in range(20000):
		# collector.p.stepSimulation()
		# time.sleep(1./240.)