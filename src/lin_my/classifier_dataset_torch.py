from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import csv
import itertools
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import angle_axis_from_quaternion, quaternion_from_angle_axis, quat2mat

# Seq2Seq dataset
######################################################
class ClassifierDataset(Dataset):
	def __init__(self, home_dir_data, data_list_dir, normal_channel=False, npoints=4096, balanced_sampling=True, split='train', with_wall=False, one_per_pair=False, use_partial_pc=False):
		self.home_dir_data = home_dir_data
		self.normal_channel = normal_channel
		self.npoints = npoints
		self.with_wall = with_wall 
		self.one_per_pair = one_per_pair
		self.use_partial_pc = use_partial_pc
		print('USING PARTIAL PC', use_partial_pc)

		self.data_dir = os.path.join(home_dir_data, 'geo_data')
		self.labels_folder_dir = os.path.join(self.data_dir, 'labels')
		self.exclude_dir = os.path.join(home_dir_data, 'exclude')
		self.cp_result_folder_dir = os.path.join(self.home_dir_data, 'dataset_cp')

		if self.use_partial_pc:
			self.partial_pc_folder_dir = os.path.join(self.home_dir_data, 'geo_data_partial_cp_pad')
			self.partial_pc_dir = {}
		all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(self.data_dir, self.exclude_dir, self.labels_folder_dir, True, with_wall=with_wall)
	
		self.all_object_name = all_object_name
		self.all_hook_name = all_hook_name
		self.n_object = len(self.all_object_name)
		self.n_hook = len(self.all_hook_name)

		# from name to numpy dir
		self.all_hook_dict = {name: {'urdf': urdf, 'pc': get_numpy_dir_from_urdf(urdf)} for name, urdf in zip(all_hook_name, all_hook_urdf)}
		self.all_object_dict = {name: {'urdf': urdf, 'pc': get_numpy_dir_from_urdf(urdf)} for name, urdf in zip(all_object_name, all_object_urdf)}

		self.pos_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result')
		self.neg_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result_neg')

		# (object_idx, hook_idx, np.array([X, 2]))

		reader = open(data_list_dir, 'r').read().splitlines()
		self.all_data = []

		self.all_result_file_names = set()

		tmp_hook_name = 'hook_wall_37'
		tmp_object_name = 'mug_61'
		ct_constraint = 40
		# positive examples
		for tmp in reader:
			result_file_name, pose_idx = '_'.join(tmp.split('_')[:-1]), tmp.split('_')[-1]
			hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
			hook_name = '{}_{}'.format(hook_cat, str(hook_id))
			object_name = '{}_{}'.format(object_cat, str(object_id))
			#if hook_name != tmp_hook_name or object_name != tmp_object_name:
			#	continue
			if not hook_name in self.all_hook_name:
				continue
			if not object_name in self.all_object_name:
				continue
			if result_file_name in self.all_result_file_names and self.one_per_pair:
				continue
			
			if self.use_partial_pc:
				partial_pc_o_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_object_partial_pc_pad.npy')
				partial_pc_h_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_hook_partial_pc_pad.npy')
				assert os.path.exists(partial_pc_h_dir) 
				assert os.path.exists(partial_pc_o_dir)
				# if not os.path.exists(partial_pc_h_dir) or not os.path.exists(partial_pc_o_dir):
					# continue
				self.partial_pc_dir[result_file_name] = {
					'hook': partial_pc_h_dir,
					'object': partial_pc_o_dir,
				}
			# if len(self.all_result_file_names) > ct_constraint:
				# break
			self.all_result_file_names.add(result_file_name)
			# assert hook_name in self.all_hook_name
			# assert object_name in self.all_object_name
			# hook_idx = self.all_hook_name.index(hook_name)
			# object_idx = self.all_object_name.index(object_name)
			# print(result_file_name, n_pose)
			self.all_data.append({
				'result_file_name': result_file_name,
				'hook_name': hook_name,
				'object_name': object_name,
				'pose_idx': int(pose_idx),
				'label': 1,
				'result_dir': os.path.join(self.pos_result_folder_dir, result_file_name + '.txt')
			})
		n_positive = len(self.all_data)
		
		# negative examples	
		if not one_per_pair:
			for result_file_name in self.all_result_file_names:
				hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
				hook_name = '{}_{}'.format(hook_cat, str(hook_id))
				object_name = '{}_{}'.format(object_cat, str(object_id))
				#if hook_name != tmp_hook_name or object_name != tmp_object_name:
			#		continue
				for pose_idx in range(10):
					self.all_data.append({
						'result_file_name': result_file_name,
						'hook_name': hook_name,
						'object_name': object_name,
						'pose_idx': int(pose_idx),
						'label': 0,
						'result_dir': os.path.join(self.neg_result_folder_dir, result_file_name + '.txt')
					})
				# break
			# self.all_data = self.all_data[:32]
		n_negative = len(self.all_data) - n_positive

		self.sampler = None

		if not one_per_pair:
			if balanced_sampling and split == 'train':
				w_pos = 1. * len(self.all_data) / n_positive
				w_neg = 1. * len(self.all_data) / n_negative
	
				weights = np.ones((len(self.all_data))) 
				weights[:n_positive] *= w_pos
				weights[n_positive:] *= w_neg
				weights = torch.DoubleTensor(weights)  
				self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
		print('n_positive', n_positive, 'n_negative', n_negative)
		
		if with_wall:
			self.pc_wall = np.load('../scripts/templates/wall/wall_small_pc.npy')[:, :3]

	
	def build_combined_pc(self, pc_o, pc_h, pose, pc_wall=None, hook_offset=None):
		pc_o_end = np.dot(pc_o, quat2mat(pose[3:]).T) + pose[:3]
		pc_o_end_w_label = np.append(pc_o_end, np.ones((self.npoints, 1)), axis=1)
		pc_h_w_label = np.append(pc_h, np.zeros((self.npoints, 1)), axis=1)

		if not (pc_wall is None):
			pc_wall_w_label = np.append(pc_wall - hook_offset, np.zeros((pc_wall.shape[0], 1)), axis=1)
		pc_combined = np.append(pc_o_end_w_label, pc_h_w_label, axis=0)

		if not (pc_wall is None):
			pc_combined = np.append(pc_combined, pc_wall_w_label, axis=0)
		return pc_combined

	def __getitem__(self, index):
		data_dict = self.all_data[index]
		result_file_name = data_dict['result_file_name']
		pc_urdf_o = self.all_object_dict[data_dict['object_name']]
		pc_urdf_h = self.all_hook_dict[data_dict['hook_name']]
		
		if not self.use_partial_pc:
			point_set_o = np.load(pc_urdf_o['pc'])
		else:
			point_set_o = np.load(self.partial_pc_dir[result_file_name]['object'])

		point_set_o = point_set_o[0:self.npoints,:]
		
		if not self.normal_channel:
			point_set_o = point_set_o[:,0:3]
			
		# load hook pc
		if not self.use_partial_pc:
			point_set_h = np.load(pc_urdf_h['pc'])
		else:
			point_set_h = np.load(self.partial_pc_dir[result_file_name]['hook'])
			
		point_set_h = point_set_h[0:self.npoints,:]
		# if not self.normal_channel:
			# point_set_h = point_set_h[:,0:3]

		pose_idx = data_dict['pose_idx']

		# load pose
		pose_o_end = load_result_file(data_dict['result_dir'])[pose_idx,-7:]
		pose_quaternion = pose_o_end[3:]
		pose_aa = angle_axis_from_quaternion(pose_quaternion)
		pose_transl_aa = np.zeros((6,))
		pose_transl_aa[:3] = pose_o_end[:3]
		pose_transl_aa[3:] = pose_aa
		pose_np = pose_transl_aa

		pc_o_end = np.dot(point_set_o, quat2mat(pose_quaternion).T) + pose_o_end[:3]
		
		pc_o_end_w_label = np.append(pc_o_end[:, :3], np.ones((self.npoints, 1)), axis=1)
		pc_h_w_label = np.append(point_set_h[:, :3], np.zeros((self.npoints, 1)), axis=1)

		hook_offset = get_hook_wall_offset(pc_urdf_h['urdf'])
		if self.with_wall:
			pc_wall_w_label = np.append(self.pc_wall - hook_offset, np.zeros((self.pc_wall.shape[0], 1)), axis=1)
		pc_combined = np.append(pc_o_end_w_label, pc_h_w_label, axis=0)

		if self.with_wall:
			pc_combined = np.append(pc_combined, pc_wall_w_label, axis=0)
		
		# from mayavi import mlab as mayalab
		# plot_pc(pc_combined)
		# mayalab.show()
	
		return {
			'pc_o': point_set_o,
			'pc_o_end': pc_o_end,
			'pc_h': point_set_h,
			'pc_combined': pc_combined,
			'pose_o': pose_o_end,
			'urdf_o': pc_urdf_o['urdf'],
			'urdf_h': pc_urdf_h['urdf'],
			'label': data_dict['label'],
			'result_file_name': result_file_name
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
			'result_file_name': result_file_name_batch
		}

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	import torch
	torch.manual_seed(2)
	
	home_dir_data = '/home/yifanyou/hang'
	# home_dir_data = '/juno/downloads/new_hang'
	cp_result_folder_dir = os.path.join(home_dir_data, 'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'test_list.txt')

	train_set = ClassifierDataset(home_dir_data, train_list_dir, False, one_per_pair=True, use_partial_pc=True)
	test_set = ClassifierDataset(home_dir_data, test_list_dir, False, split='test', one_per_pair=True, use_partial_pc=True)

	# dataset = CPS2Dataset('/scr1/yifan/hang', False)
	
	print('len train', len(train_set))
	print('len test', len(test_set))
	one_data = train_set[0]
	one_data = train_set[1]
	one_data = train_set[2]
	one_data = train_set[3]
	one_data = train_set[4]
	one_data = train_set[8]
	train_loader = DataLoader(train_set, batch_size=32, sampler=train_set.sampler,
					num_workers=1, collate_fn=ClassifierDataset.pad_collate_fn_for_dict)
	test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
					num_workers=1, collate_fn=ClassifierDataset.pad_collate_fn_for_dict)
	
	# import time
	# t_a = time.time()
	# for i, one_data in enumerate(train_loader):
	# 	print()
	# 	print('pc o', one_data['input1'].shape)
	# 	print('pc h', one_data['input2'].shape)
	# 	print('pc combined', one_data['input3'].shape)
	# 	print('pose o end', one_data['output4'].shape)

	# 	print(one_data['input3'].shape)
	# 	print(np.sum(one_data['label']))

	# 	# print(one_data['urdf_o'])
	# 	# print(np.nonzero(one_data['output4']))
	# 	if i > 20:
	# 		break
	# print(time.time() - t_a)
