from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import csv
import itertools
import numpy as np
import pickle
from scipy.spatial import KDTree, cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import *

class MyDataset(Dataset):
	def __init__(self, home_dir_data, data_list_dir, is_train=True, normal_channel=True, npoints=4096, balanced_sampling=True, with_wall=False, add_pene_data=True, use_partial_pc=True, args=None):
		self.home_dir_data = home_dir_data
		self.normal_channel = normal_channel
		self.npoints = npoints
		self.with_wall = with_wall
		self.balanced_sampling = balanced_sampling
		self.overfit = args.overfit
		if is_train:
			self.split = 'train'
		else:
			self.split = 'test'
		self.use_partial_pc = use_partial_pc
		print('USING PARTIAL PC', use_partial_pc)

		self.data_dir = os.path.join(home_dir_data, 'geo_data')
		self.collection_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result')

		self.collection_result_neg_folder_dir = os.path.join(self.home_dir_data, 'collection_result_neg')
		data_list_neg_dir = os.path.join(self.collection_result_neg_folder_dir, 'labels', 'all_list.txt')
		
		if True:
			self.pose_result_more_folder_dir = os.path.join(self.home_dir_data, 'collection_result_more')
			self.pose_result_more_takeoff_dict_dir = os.path.join(self.home_dir_data, 'collection_result_more_seq', 'labels', 'all_dict.json')
			data_list_more_pose_dir = os.path.join(self.pose_result_more_folder_dir, 'labels', 'all_list.txt')

		if add_pene_data:
			self.pos_pene_big_output_dir = os.path.join(home_dir_data, 'collection_result_pene_big_pos_new')
			data_list_pos_pene_big_dir = os.path.join(self.pos_pene_big_output_dir, 'labels', 'all_list.txt')

						
		if self.use_partial_pc:
			self.partial_pc_folder_dir = os.path.join(self.home_dir_data, 'geo_data_partial_cp_pad')
			self.partial_pc_dir = {}	
			
		all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(self.data_dir)
		# from name to numpy dir
		self.all_hook_dict = {name: {'urdf': urdf, 'pc': get_numpy_dir_from_urdf(urdf)} for name, urdf in zip(all_hook_name, all_hook_urdf)}
		self.all_object_dict = {name: {'urdf': urdf, 'pc': get_numpy_dir_from_urdf(urdf)} for name, urdf in zip(all_object_name, all_object_urdf)}

		self.all_data = []
		self.all_result_file_names = set()
		self.object_cat_dict_pos = {}
		self.object_cat_dict_neg = {}

		self.all_object_name = all_object_name
		self.all_hook_name = all_hook_name
		self.n_object = len(self.all_object_name)
		self.n_hook = len(self.all_hook_name)

		# positive data
		reader = load_txt(data_list_dir)
		self.load_from_reader(reader, 'ori', self.collection_result_folder_dir, True)

		# more_pose_filter_dict = load_json(self.pose_result_more_takeoff_dict_dir)
		reader_more_pose = load_txt(data_list_more_pose_dir)
		self.load_from_reader(reader_more_pose, 'more_pose', self.pose_result_more_folder_dir, True, filter_dict=None)

		# negative data
		reader_neg = load_txt(data_list_neg_dir)
		self.load_from_reader(reader_neg, 'neg_pose', self.collection_result_neg_folder_dir, False)

		if add_pene_data:
			reader_big_data_pos = load_txt(data_list_pos_pene_big_dir)
			self.load_from_reader(reader_big_data_pos, 'pene_data_pos', self.pos_pene_big_output_dir, False)

		if self.overfit:
			np.random.shuffle(self.all_data)
			self.all_data = self.all_data[:args.batch_size]
			print([tmp['result_file_name'] for tmp in self.all_data])

		self.n_positive = 0
		self.n_negative = 0
		self.n_neg_pene = 0
		self.n_neg_non_pene = 0
		self.neg_pene_idx = []
		self.neg_non_pene_idx = []
		
		for ii in range(len(self.all_data)):
			if self.all_data[ii]['label'] == 0:
				if self.all_data[ii]['dataset_label'] == 'pene_data_pos':
					self.n_neg_pene += 1
					self.neg_pene_idx.append(ii)
				elif self.all_data[ii]['dataset_label'] == 'neg_pose':
					self.n_neg_non_pene += 1
					self.neg_non_pene_idx.append(ii)
				self.n_negative += 1
			else:
				self.n_positive += 1

		from collections import Counter
		total_n_pose_pos = 0
		total_n_pose_neg = 0
		total_n_pair = 0
		for object_cat in self.object_cat_dict_neg:
			if not object_cat in self.object_cat_dict_pos:
				cat_n_pose_pos = 0
				cat_n_pair_pos = 0
			else:
				cat_n_pose_pos = len(self.object_cat_dict_pos[object_cat])
				cat_n_pair_pos = len(Counter(self.object_cat_dict_pos[object_cat]).keys())
			if not object_cat in self.object_cat_dict_neg:
				cat_n_pose_neg = 0
				cat_n_pair_neg = 0
			else:
				cat_n_pose_neg = len(self.object_cat_dict_neg[object_cat])
				cat_n_pair_neg = len(Counter(self.object_cat_dict_neg[object_cat]).keys())
			cat_n_pair = max(cat_n_pair_pos, cat_n_pair_neg)
			print(object_cat, 'n pose', cat_n_pose_pos + cat_n_pose_neg, 'n hook-object pair', cat_n_pair, 'pos', cat_n_pose_pos / cat_n_pair, 'neg', cat_n_pose_neg / cat_n_pair)
			total_n_pose_pos += cat_n_pose_pos
			total_n_pose_neg += cat_n_pose_neg
			total_n_pair += cat_n_pair
		print('total', 'n pose', total_n_pose_pos + total_n_pose_neg, 'n_pair', total_n_pair, 'pos pose', total_n_pose_pos / total_n_pair, 'neg pose', total_n_pose_neg / total_n_pair)

		print('n_positive', self.n_positive, 'n_negative', self.n_negative)
		
	def load_from_reader(self, reader, dataset_label, dataset_dir, label_is_true, from_buffer=False, filter_dict=None):
		ori = dataset_label == 'ori'
		tmp_ct = 0
		for tmp in reader:
			result_file_name, pose_idx = '_'.join(tmp.split('_')[:-1]), tmp.split('_')[-1]
			
			if not(filter_dict is None):
				if not (tmp in filter_dict):
					continue
				if not filter_dict[tmp]:
					continue

			if (not ori) and (not result_file_name in self.all_result_file_names):
				continue 
			hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
			hook_name = '{}_{}'.format(hook_cat, str(hook_id))
			object_name = '{}_{}'.format(object_cat, str(object_id))

			if not hook_name in self.all_hook_name:
				continue
			if not object_name in self.all_object_name:
				continue
			hook_idx = self.all_hook_name.index(hook_name)
			object_idx = self.all_object_name.index(object_name)

			if ori and self.use_partial_pc:
				partial_pc_o_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_object_partial_pc_pad.npy')
				partial_pc_h_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_hook_partial_pc_pad.npy')

				partial_pc_o_idx_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_object_partial_pc_pad_idx.npy')
				partial_pc_h_idx_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_hook_partial_pc_pad_idx.npy')
				if not os.path.exists(partial_pc_h_dir) or not os.path.exists(partial_pc_o_dir):
					continue
				self.partial_pc_dir[result_file_name] = {
					'hook': partial_pc_h_dir,
					'object': partial_pc_o_dir,
					'hook_idx': partial_pc_h_idx_dir,
					'object_idx': partial_pc_o_idx_dir
				}
			if ori:
				self.all_result_file_names.add(result_file_name)
			tmp_data_dict = {
				'dataset_label': dataset_label,
				'result_file_name': result_file_name,
				'hook_name': hook_name,
				'object_name': object_name,
				'pose_idx': int(pose_idx),
				'label': 1 if label_is_true else 0,
				'from_buffer': 1 if from_buffer else 0,
				'result_dir': os.path.join(dataset_dir, result_file_name + '.txt'),
			}
			if label_is_true:
				object_cat_dict = self.object_cat_dict_pos
			else:
				object_cat_dict = self.object_cat_dict_neg

			if not object_cat in object_cat_dict:
				object_cat_dict[object_cat] = [result_file_name] 
			else:
				object_cat_dict[object_cat].append(result_file_name)

			self.all_data.append(tmp_data_dict)
			tmp_ct += 1
			if self.overfit:
				if tmp_ct > 100:
					break
			
	def build_sampler(self):
		assert len(self.all_data) == self.n_positive + self.n_negative
		assert self.n_negative == self.n_neg_non_pene + self.n_neg_pene
		self.sampler = None
		if self.balanced_sampling and self.split == 'train':
			all_data_labels = [tmp['label'] for tmp in self.all_data]

			w_pos = 1. * len(self.all_data) / self.n_positive * 2
			w_neg_pene = 1. * len(self.all_data) / self.n_neg_pene
			w_neg_non_pene = 1. * len(self.all_data) / self.n_neg_non_pene

			weights = np.ones((len(self.all_data))) 
			weights[np.where(np.array(all_data_labels) == 1)] *= w_pos 
			weights[np.array(self.neg_pene_idx)] *= w_neg_pene
			weights[np.array(self.neg_non_pene_idx)] *= w_neg_non_pene
			
			weights = torch.DoubleTensor(weights)  
			self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

	def __getitem__(self, index):
		data_dict = self.all_data[index]
		result_file_name = data_dict['result_file_name']
		from_buffer = data_dict['from_buffer']

		pc_urdf_o = self.all_object_dict[data_dict['object_name']]
		pc_urdf_h = self.all_hook_dict[data_dict['hook_name']]
		
		if not self.use_partial_pc:
			point_set_o = np.load(pc_urdf_o['pc'])
		else:
			point_set_o = np.load(self.partial_pc_dir[result_file_name]['object'])

		if not self.normal_channel:
			point_set_o = point_set_o[:,:3]
			
		# load hook pc
		if not self.use_partial_pc:
			point_set_h = np.load(pc_urdf_h['pc'])
		else:
			point_set_h = np.load(self.partial_pc_dir[result_file_name]['hook'])

		point_set_h = point_set_h[0:self.npoints,:]
		if not self.normal_channel:
			point_set_h = point_set_h[:,0:3]

		if from_buffer == 0:
			pose_idx = data_dict['pose_idx']

			# load pose
			pose_o_end = load_result_file(data_dict['result_dir'])[pose_idx,-7:]

		elif from_buffer == 1:
			pose_o_end = data_dict['pose_o_end']

		pc_combined = create_pc_combined(point_set_o, point_set_h, pose_o_end[:3], pose_o_end[3:], normal=self.normal_channel)

		# if data_dict['dataset_label'] == 'neg_pose':
		# 	obj_pc = pc_combined
		# 	from mayavi import mlab as mayalab 
		# 	plot_pc(obj_pc)
		# 	print(data_dict['dataset_label'])
		# 	mayalab.show()
		# mayalab.quiver3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], obj_pc[:, 3], obj_pc[:, 4], obj_pc[:, 5], scale_factor=0.005)
		# # if data_dict['label'] == 1:
		# print('label', data_dict['label'], data_dict['object_name'], data_dict['hook_name'])
		# print(pose_o_end, obj_pc.shape)
		# # mayalab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], scale_factor=0.005)
		# print('show it')
		# mayalab.show()

		return {
			'pc_o': point_set_o,
			# 'pc_o_end': pc_o_end,
			'pc_h': point_set_h[:, :3],
			'pc_combined': pc_combined,
			'pose_o': pose_o_end,
			'urdf_o': pc_urdf_o['urdf'],
			'urdf_h': pc_urdf_h['urdf'],
			'label': data_dict['label'],
			'dataset_label': data_dict['dataset_label'],
			'result_file_name': result_file_name,
		} 

	def __len__(self):
		return len(self.all_data)

	@staticmethod
	def pad_collate_fn_for_dict(batch):
		pc_o_batch = [d['pc_o'] for d in batch]
		pc_o_batch = np.stack(pc_o_batch, axis=0)

		pc_h_batch = [d['pc_h'] for d in batch]
		pc_h_batch = np.stack(pc_h_batch, axis=0)

		pc_combined_batch = [d['pc_combined'] for d in batch]
		pc_combined_batch = np.stack(pc_combined_batch, axis=0)

		pose_o_batch = [d['pose_o'] for d in batch]
		pose_o_batch = np.stack(pose_o_batch, axis=0)
		
		urdf_o_batch = [d['urdf_o'] for d in batch]
		urdf_h_batch = [d['urdf_h'] for d in batch]

		result_file_name_batch = [d['result_file_name'] for d in batch]

		label_batch = np.array([d['label'] for d in batch])
		dataset_label_batch = [d['dataset_label'] for d in batch]

		return {
			'input1': pc_o_batch, # (batch_size, 4096, 3 or 6) np.float64
			'input2': pc_h_batch, # (batch_size, 4096, 3 or 6) np.float64
			'input3': pc_combined_batch, #(batch_size, 4096*2, 4)

			'output1': None, # (batch_size, [varies], 2) list of np.int64 array
			'output2': None, # (batch_size, [varies], 2) list of np.int64 array
			'output3': None, # (batch_size, 4096, 4096) np.bool_
			'output4': pose_o_batch, # (batch_size, 7) np.float64
			'urdf_o': urdf_o_batch,
			'urdf_h': urdf_h_batch,
			'label': label_batch,
			'dataset_label': dataset_label_batch,
			'result_file_name': result_file_name_batch
		}

if __name__ == "__main__":
	import argparse
	from torch.utils.data import DataLoader
	import torch
	torch.manual_seed(2)
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument('--bohg4', action='store_true')

	parser.add_argument('--pointset_dir', default='/scr2/')
	parser.add_argument('--restrict_object_cat', default='')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--data_one_pose', action='store_true')
	parser.add_argument('--data_vary_scale', action='store_true')
	parser.add_argument('--data_more_pose', action='store_true')
	parser.add_argument('--data_vary_scale_more_pose', action='store_true')
	parser.add_argument('--cp', action='store_true')
	parser.add_argument('--with_min_dist_pc', action='store_true')
	parser.add_argument('--batch_size', type=int, default=32)

	args = parser.parse_args()
	if args.bohg4:
		args.home_dir_data = '/scr1/yifan/hang'

	home_dir_data = args.home_dir_data
	cp_result_folder_dir= os.path.join(home_dir_data,'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir,'labels','train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir,'labels','test_list.txt')


	train_set = MyDataset(home_dir_data, train_list_dir, True, use_partial_pc=True, args=args)
	test_set = MyDataset(home_dir_data, test_list_dir, False, use_partial_pc=True, args=args)


	print('len train', len(train_set), train_set.n_positive, train_set.n_negative)
	print('len test', len(test_set))
	train_set.build_sampler()
	train_loader = DataLoader(train_set, batch_size=32, sampler=train_set.sampler,
					num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
					num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	import time
	t_a = time.time()
	for i, one_data in enumerate(train_loader):
		# print()
		# print('pc o', one_data['input1'].shape)
		# print('pc h', one_data['input2'].shape)
		# print('pc combined', one_data['input3'].shape)
		# print('pose o end', one_data['output4'].shape)

		# print(one_data['input3'].shape)

		tmp_ct_pene = 0
		tmp_ct_non_pene = 0
		for j, tmp in enumerate(one_data['dataset_label']):
			if 'pene' in tmp:
				assert one_data['label'][j] == 0
				tmp_ct_pene += 1
			elif 'neg_pose' in tmp:
				assert one_data['label'][j] == 0
				tmp_ct_non_pene += 1
		print('pos {} neg {} neg pene {} neg non pene {}'.format(np.sum(one_data['label']), len(one_data['label']) - np.sum(one_data['label']), tmp_ct_pene, tmp_ct_non_pene))

		# print(one_data['urdf_o'])
		# print(np.nonzero(one_data['output4']))
		if i > 20:
			break

	print(time.time() - t_a)
	pass
