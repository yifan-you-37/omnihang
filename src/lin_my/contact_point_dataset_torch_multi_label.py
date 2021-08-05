from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import csv
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import angle_axis_from_quaternion, quaternion_from_angle_axis

# Seq2Seq dataset
######################################################
class MyDataset(Dataset):
	def __init__(self, home_dir_data, data_list_dir, is_train=True, normal_channel=True, npoints=4096, use_partial_pc=True, overfit=False, restrict_object_cat=None):
		self.home_dir_data = home_dir_data
		self.normal_channel = normal_channel
		self.npoints = npoints
		self.use_partial_pc = use_partial_pc
		self.is_train = is_train
		print('USING PARTIAL PC', use_partial_pc)

		if self.use_partial_pc:
			self.data_dir = os.path.join(home_dir_data, 'geo_data')
			
			self.partial_pc_folder_dir = os.path.join(self.home_dir_data, 'geo_data_partial_cp_pad')
			self.partial_pc_dir = {}
		else:
			self.data_dir = os.path.join(home_dir_data, 'geo_data_old')

		self.labels_folder_dir = os.path.join(self.data_dir, 'labels')
		self.exclude_dir = os.path.join(home_dir_data, 'exclude')

		all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(self.data_dir, self.exclude_dir, self.labels_folder_dir, True, with_wall=False)
	
		self.all_object_name = all_object_name
		self.all_hook_name = all_hook_name
		self.n_object = len(self.all_object_name)
		self.n_hook = len(self.all_hook_name)

		# from name to numpy dir
		if not use_partial_pc:
			self.all_hook_np_dict = {i: get_numpy_dir_from_urdf(urdf) for i, urdf in enumerate(all_hook_urdf)}
			self.all_object_np_dict = {i: get_numpy_dir_from_urdf(urdf) for i, urdf in enumerate(all_object_urdf)}

		if use_partial_pc:
			self.cp_result_folder_dir = os.path.join(self.home_dir_data, 'dataset_cp_partial')
			self.pose_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result')
		else:
			self.cp_result_folder_dir = os.path.join(self.home_dir_data, 'dataset_cp')
			self.pose_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result_old')

		reader = open(data_list_dir, 'r').read().splitlines()
		self.all_data = []
		self.all_data_dict = {}
		all_result_file_name = set()

		object_cat_dict = {}
		data_ct = 0

		np.random.shuffle(reader)
		
		for tmp in reader:
			result_file_name, pose_idx = '_'.join(tmp.split('_')[:-1]), tmp.split('_')[-1]
			hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
			hook_name = '{}_{}'.format(hook_cat, str(hook_id))
			object_name = '{}_{}'.format(object_cat, str(object_id))
			if (not restrict_object_cat is None) and (not restrict_object_cat == '') and object_cat != restrict_object_cat:
				continue
			if not object_cat in object_cat_dict:
				object_cat_dict[object_cat] = [result_file_name]
			else:
				object_cat_dict[object_cat].append(result_file_name)
			
			# TODO
			# if result_file_name in all_result_file_name:
				# continue
			# all_result_file_name.add(result_file_name)

			if not hook_name in self.all_hook_name:
				continue
			if not object_name in self.all_object_name:
				continue
			hook_idx = self.all_hook_name.index(hook_name)
			object_idx = self.all_object_name.index(object_name)

			if self.use_partial_pc:
				partial_pc_o_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_object_partial_pc_pad.npy')
				partial_pc_h_dir = os.path.join(self.partial_pc_folder_dir, result_file_name + '_hook_partial_pc_pad.npy')
				# if not os.path.exists(partial_pc_h_dir) or not os.path.exists(partial_pc_o_dir):
					# continue
				self.partial_pc_dir[result_file_name] = {
					'hook': partial_pc_h_dir,
					'object': partial_pc_o_dir,
				}
				# cp_idx_o_dir = os.path.join(self.cp_result_folder_dir, result_file_name + '_' + str(pose_idx) +  '_idx_object.npy')
				# cp_idx_h_dir = os.path.join(self.cp_result_folder_dir, result_file_name + '_' + str(pose_idx) +  '_idx_hook.npy')
				# if not os.path.exists(cp_idx_o_dir) or not os.path.exists(cp_idx_h_dir):
				# 	continue
				# assert os.path.exists(cp_idx_o_dir)
				# assert os.path.exists(cp_idx_h_dir)

			# print(result_file_name, n_pose)

			tmp_data_dict = {
				'cp_result_file_name': result_file_name + '_' + str(pose_idx),
				'object_idx': object_idx,
				'hook_idx': hook_idx,
				'pose_idx': int(pose_idx),
			}
			if not (result_file_name in self.all_data_dict):
				self.all_data_dict[result_file_name] = [tmp_data_dict]
			else:
				# if self.is_train:
					# continue
				self.all_data_dict[result_file_name].append(tmp_data_dict)
			data_ct += 1
			
			if overfit:
				if data_ct == 1:
					break
		self.all_data = list(self.all_data_dict.keys())
		self.all_data.sort()

			# np.random.shuffle(self.all_data)
			# self.all_data = self.all_data[:len(self.all_data)//2]
		for object_cat in object_cat_dict:
			print(object_cat, len(object_cat_dict[object_cat]))
	def __getitem__(self, index):
		result_file_name = self.all_data[index]
		data_dict_arr = self.all_data_dict[result_file_name]
		data_dict = data_dict_arr[0]
		
		if not self.use_partial_pc:
			point_set_o = np.load(self.all_object_np_dict[data_dict['object_idx']])
		else:
			point_set_o = np.load(self.partial_pc_dir[result_file_name]['object'])

		point_set_o = point_set_o[0:self.npoints,:]
		if not self.normal_channel:
			point_set_o = point_set_o[:,0:3]

		# load hook pc
		if not self.use_partial_pc:
			point_set_h = np.load(self.all_hook_np_dict[data_dict['hook_idx']])
		else:
			point_set_h = np.load(self.partial_pc_dir[result_file_name]['hook'])
		point_set_h = point_set_h[0:self.npoints,:]
		if not self.normal_channel:
			point_set_h = point_set_h[:,0:3]

		all_pose_o = np.zeros((len(data_dict_arr), 6))
		pose_result_dir = os.path.join(self.pose_result_folder_dir, result_file_name + '.txt')
		pose_result = load_result_file(pose_result_dir)
		for ii, data_dict in enumerate(data_dict_arr):
			cp_result_file_name = data_dict['cp_result_file_name']
			pose_idx = data_dict['pose_idx']

			# load cp arr
			# cp_idx_o_dir = os.path.join(self.cp_result_folder_dir, cp_result_file_name +  '_idx_object.npy')
			# cp_idx_h_dir = os.path.join(self.cp_result_folder_dir, cp_result_file_name +  '_idx_hook.npy')

			# cp_idx_o = np.load(cp_idx_o_dir)
			# cp_idx_h = np.load(cp_idx_h_dir)

			# load pose
			pose_o_end = pose_result[pose_idx,-7:]
			pose_quaternion = pose_o_end[3:]
			pose_aa = angle_axis_from_quaternion(pose_quaternion)
			pose_transl_aa = np.zeros((6,))
			pose_transl_aa[:3] = pose_o_end[:3]
			pose_transl_aa[3:] = pose_aa
			pose_np = pose_transl_aa
			all_pose_o[ii] = pose_np
			# plot_pc(point_set_h)
			# plot_pc(point_set_h[cp_idx_h], color=[1, 0, 0])
			# from mayavi import mlab as mayalab 
			# mayalab.show()
			
		return {
			'pc_o':point_set_o,
			'pc_h':point_set_h, 
			# 'cp_idx_o':cp_idx_o,
			# 'cp_idx_h':cp_idx_h,
			'cp_idx_o':None,
			'cp_idx_h':None,
			'pose_o':all_pose_o,
			'n_pose': all_pose_o.shape[0],
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

		cp_idx_o_batch = [d['cp_idx_o'] for d in batch]
		cp_idx_h_batch = [d['cp_idx_h'] for d in batch]

		pose_o_batch = [d['pose_o'] for d in batch]
		n_pose = [d['n_pose'] for d in batch]
		max_n_pose = max(n_pose)
		pose_o_batch = list(map(lambda x : pad_np(x, pad=max_n_pose, dim=0), pose_o_batch))
		pose_o_batch = np.stack(pose_o_batch, axis=0)
		# pose_o_batch = np.stack(pose_o_batch, axis=0)
		
		result_file_name = [d['result_file_name'] for d in batch]

		return {
			'input1': pc_o_batch, # (batch_size, 4096, 3 or 6) np.float64
			'input2': pc_h_batch, # (batch_size, 4096, 3 or 6) np.float64
			'output1': cp_idx_o_batch, # (batch_size, [varies], 2) list of np.int64 array
			'output2': cp_idx_h_batch, # (batch_size, [varies], 2) list of np.int64 array
			'output3': None, # (batch_size, 4096, 4096) np.bool_
			'output4': pose_o_batch, # (batch_size, 7) np.float64
			'result_file_name': result_file_name,
			'n_pose': n_pose
		}

def pad_np(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return np.concatenate([vec, np.zeros(pad_size)], axis=dim)

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	import torch
	torch.manual_seed(2)
	
	home_dir_data = '/scr1/yifan/hang'
	cp_result_folder_dir= os.path.join(home_dir_data,'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir,'labels','train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir,'labels','test_list.txt')

	train_set = MyDataset(home_dir_data, train_list_dir, True, use_partial_pc=True)
	test_set = MyDataset(home_dir_data, test_list_dir, True, use_partial_pc=True)

	print('len train', len(train_set))
	print('len test',len(test_set))
	train_loader = DataLoader(train_set, batch_size=2, shuffle=True,
					num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	# test_loader = DataLoader(test_set, batch_size=2, shuffle=True,
					# num_workers=2, collate_fn=CPDataset.pad_collate_fn_for_dict)
	import time
	t_a = time.time()
	for i, one_data in enumerate(train_loader):
		print(i)
		print('pc o', one_data['input1'].shape)
		print('pc h', one_data['input2'].shape)
		print('pose o end', one_data['output4'].shape)
		print('pose o end', one_data['output4'])
		print('cp_idx_o', len(one_data['output2']))
		print('cp_idx_h', len(one_data['output1']))
		if i > 10:
			break
	print(time.time() - t_a)
	pass
