from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import csv
import itertools
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import *
from collision_helper import *
class MyDataset(Dataset):
	def __init__(self, home_dir_data, data_list_dir, is_train=True, normal_channel=True, npoints=4096, use_partial_pc=True, use_fcl=True, load_pose=False, one_pose_per_pair=True, split_n=None, split_id=None, args=None):
		self.home_dir_data = home_dir_data
		self.normal_channel = normal_channel
		self.npoints = npoints
		self.overfit = args.overfit
		self.use_fcl = use_fcl
		self.load_pose = load_pose
		self.one_pose_per_pair = one_pose_per_pair
		if is_train:
			self.split = 'train'
		else:
			self.split = 'test'
		self.use_partial_pc = use_partial_pc
		print('USING PARTIAL PC', use_partial_pc)

		self.data_dir = os.path.join(home_dir_data, 'geo_data')
						
		if self.use_partial_pc:
			self.partial_pc_folder_dir = os.path.join(self.home_dir_data, 'geo_data_partial_cp_pad')
			self.partial_pc_dir = {}	
			
		all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(self.data_dir)

		# all_hook_name = [all_hook_name[0]]
		# all_object_name = all_object_name[:10]
		# all_hook_urdf = [all_hook_urdf[0]]
		# all_object_urdf = all_object_urdf[:10]
		
		# from name to numpy dir
		self.all_hook_dict = {name: {'urdf': urdf, 'pc': get_numpy_dir_from_urdf(urdf)} for name, urdf in zip(all_hook_name, all_hook_urdf)}
		self.all_object_dict = {name: {'urdf': urdf, 'pc': get_numpy_dir_from_urdf(urdf)} for name, urdf in zip(all_object_name, all_object_urdf)}


		#fcl models initialize
		self.fcl_hook_dict = {name: (fcl_half_load_urdf(urdf) if self.use_fcl else None) for name, urdf in zip(all_hook_name, all_hook_urdf)}
		self.fcl_object_dict = {name: (fcl_half_load_urdf(urdf) if self.use_fcl else None) for name, urdf in zip(all_object_name, all_object_urdf)}

		self.all_data = []
		self.all_result_file_names = set()
		self.object_cat_dict = {}

		self.all_object_name = all_object_name
		self.all_hook_name = all_hook_name
		self.n_object = len(self.all_object_name)
		self.n_hook = len(self.all_hook_name)
		
		# positive data
		reader = load_txt(data_list_dir)
		self.collection_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result')
		self.load_from_reader(reader, 'ori', self.collection_result_folder_dir)
		
		if (not split_n is None) and (split_n > 0):
			assert split_id >= 0
			assert split_id < split_n
			# split_len = len(self.all_data) // split_n
			# start_id = (split_id * split_len)

			# if split_id == split_n - 1:
			# 	self.all_data = self.all_data[start_id:]
			# else:
			# 	self.all_data = self.all_data[start_id:start_id+split_len]
			self.all_data = self.all_data[split_id::split_n]

			# assert split_id >= 0
			# split_len = len(self.all_data) // split_n + 1
			# start_id = (split_id * split_len)
			# self.all_data = self.all_data[start_id:start_id+split_len]
		for ii, tmp in enumerate(self.all_data):
			print(tmp['result_file_name'])
			if ii >= 10:
				break
		if self.overfit:
			np.random.shuffle(self.all_data)
			# filter_txt_dir = os.path.join(args.home_dir_data, 'dataset_cp', 'labels', 'result_file_name_s3_both.txt')
			# filter_set = set(load_txt(filter_txt_dir))

			# tmp_data = []
			# for tmp in self.all_data:
			# 	if tmp['result_file_name'] in filter_set:
			# 		tmp_data.append(tmp)
			# 		if len(tmp_data) == 1000:
			# 			break
						
			# self.all_data = tmp_data
			self.all_data = self.all_data[:50]
			print([tmp['result_file_name'] for tmp in self.all_data])
			for tmp in self.all_data:
				print('{}_{}'.format(tmp['result_file_name'], tmp['pose_idx']))

		self.result_file_name_data_idx = {}
		for ii in range(len(self.all_data)):
			self.result_file_name_data_idx[self.all_data[ii]['result_file_name']] = ii
		print('number of pairs loaded', len(self.all_data))
		if len(self.all_data) > 0:
			print_object_cat_dict(self.object_cat_dict)
		
	def load_from_reader(self, reader, dataset_label, dataset_dir, filter_dict=None):
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

			if self.one_pose_per_pair:
				# one data entry per object_hook_pair
				if ori and (result_file_name in self.all_result_file_names):
					continue

			hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
			hook_name = '{}_{}'.format(hook_cat, str(hook_id))
			object_name = '{}_{}'.format(object_cat, str(object_id))

			if not hook_name in self.all_hook_name:
				continue
			if not object_name in self.all_object_name:
				continue
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
				}
			if ori:
				self.all_result_file_names.add(result_file_name)
				
			tmp_data_dict = {
				'dataset_label': dataset_label,
				'result_file_name': result_file_name,
				'hook_name': hook_name,
				'object_name': object_name,
				'pose_idx': int(pose_idx),
				'result_dir': os.path.join(dataset_dir, result_file_name + '.txt'),
			}

			if not object_cat in self.object_cat_dict:
				self.object_cat_dict[object_cat] = [result_file_name] 
			else:
				self.object_cat_dict[object_cat].append(result_file_name)

			self.all_data.append(tmp_data_dict)
			tmp_ct += 1
			
	def __getitem__(self, index):
		data_dict = self.all_data[index]
		result_file_name = data_dict['result_file_name']

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

		pose_o_end = None
		pc_combined = None
		if self.load_pose:
			pose_idx = data_dict['pose_idx']
			pose_o_end = load_result_file(data_dict['result_dir'])[pose_idx,-7:]
			pc_combined = create_pc_combined(point_set_o, point_set_h, pose_o_end[:3], pose_o_end[3:], normal=self.normal_channel)

		# print(result_file_name)
		# from mayavi import mlab as mayalab 
		# plot_pc(point_set_o)
		# mayalab.show()
		# plot_pc(point_set_h)
		# mayalab.show()

		# if self.use_fcl:
			# fcl_hook_model = self.fcl_hook_dict[data_dict['hook_name']]
			# fcl_object_model = self.fcl_object_dict[data_dict['object_name']]

		return {
			'pc_o': point_set_o,
			'pc_h': point_set_h[:, :3],
			'urdf_o': pc_urdf_o['urdf'],
			'urdf_h': pc_urdf_h['urdf'],
			'result_file_name': result_file_name,
			'hook_name': data_dict['hook_name'],
			'object_name': data_dict['object_name'],
			'pc_combined': pc_combined,
			'pose_o': pose_o_end,
			# 'fcl_hook_model': fcl_hook_model if self.use_fcl else None,
			# 'fcl_object_model': fcl_object_model if self.use_fcl else None,

		} 

	def __len__(self):
		return len(self.all_data)

	@staticmethod
	def pad_collate_fn_for_dict(batch):
		pc_o_batch = [d['pc_o'] for d in batch]
		pc_o_batch = np.stack(pc_o_batch, axis=0)

		pc_h_batch = [d['pc_h'] for d in batch]
		pc_h_batch = np.stack(pc_h_batch, axis=0)
		
		urdf_o_batch = [d['urdf_o'] for d in batch]
		urdf_h_batch = [d['urdf_h'] for d in batch]

		result_file_name_batch = [d['result_file_name'] for d in batch]

		# fcl_hook_model_batch = [d['fcl_hook_model'] for d in batch]
		# fcl_object_model_batch = [d['fcl_object_model'] for d in batch]
		pc_combined_batch = [d['pc_combined'] for d in batch]
		pc_combined_batch = np.stack(pc_combined_batch, axis=0)

		pose_o_batch = [d['pose_o'] for d in batch]
		pose_o_batch = np.stack(pose_o_batch, axis=0)

		hook_name_batch = [d['hook_name'] for d in batch]
		object_name_batch = [d['object_name'] for d in batch]
		return {
			'input1': pc_o_batch, # (batch_size, 4096, 3 or 6) np.float64
			'input2': pc_h_batch, # (batch_size, 4096, 3 or 6) np.float64
			'input3': pc_combined_batch, #(batch_size, 4096*2, 4)
			'output4': pose_o_batch, # (batch_size, 7) np.float64
			'urdf_o': urdf_o_batch,
			'urdf_h': urdf_h_batch,
			'result_file_name': result_file_name_batch,
			'hook_name': hook_name_batch,
			'object_name': object_name_batch
			# 'fcl_hook_model': fcl_hook_model_batch,
			# 'fcl_object_model': fcl_object_model_batch,

		}

if __name__ == "__main__":
	import argparse
	from torch.utils.data import DataLoader
	import torch
	torch.manual_seed(2)
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument('--bohg4', action='store_true')
	parser.add_argument('--train_list', default='train_list')
	parser.add_argument('--test_list', default='test_list')

	parser.add_argument('--pointset_dir', default='/scr2/')
	parser.add_argument('--restrict_object_cat', default='')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--data_one_pose', action='store_true')
	parser.add_argument('--data_vary_scale', action='store_true')
	parser.add_argument('--one_pose_per_pair', action='store_true')
	parser.add_argument('--cp', action='store_true')
	parser.add_argument('--with_min_dist_pc', action='store_true')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--full_pc', action='store_true')
	parser.add_argument('--no_fcl', action='store_true')
	parser.add_argument('--parallel_n', default=-1, type=int)
	parser.add_argument('--parallel_id', default=-1, type=int)

	args = parser.parse_args()
	if args.bohg4:
		args.home_dir_data = '/scr1/yifan/hang'

	home_dir_data = args.home_dir_data
	cp_result_folder_dir= os.path.join(home_dir_data,'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir,'labels','{}_list.txt'.format(args.train_list))
	test_list_dir = os.path.join(cp_result_folder_dir,'labels','{}.txt'.format(args.test_list))


	# train_set = MyDataset(home_dir_data, train_list_dir, True, use_partial_pc=(not args.full_pc), use_fcl=(not args.no_fcl), load_pose=True, args=args)
	test_set = MyDataset(home_dir_data, test_list_dir, False, use_partial_pc=(not args.full_pc), use_fcl=(not args.no_fcl), one_pose_per_pair=args.one_pose_per_pair, split_n=args.parallel_n, split_id=args.parallel_id, args=args)


	# print('len train', len(train_set))
	print('len test', len(test_set))
	# train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
					# num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
					num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	import time
	t_a = time.time()
	for i, one_data in enumerate(test_loader):
		# tmp = [train_set.fcl_hook_dict[name] for name in one_data['hook_name']]
		print(tmp)
		print(one_data['input3'].shape)
		break
		# print()
		# print('pc o', one_data['input1'].shape)
		# print('pc h', one_data['input2'].shape)
		# print('pc combined', one_data['input3'].shape)
		# print('pose o end', one_data['output4'].shape)

		# print(one_data['input3'].shape)
		# print(one_data['result_file_name'][0])

		# print(one_data['urdf_o'])
		# print(np.nonzero(one_data['output4']))
		# if i > 20:
			# break

	print(time.time() - t_a)
	pass
