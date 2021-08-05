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
	
class MyDataset(Dataset):
	def __init__(self, home_dir_data, data_list_dir, is_train=True, normal_channel=True, npoints=4096, use_partial_pc=True, cp_s2a_soft=False, cp_s2b_soft=False, args=None):
		self.home_dir_data = home_dir_data
		self.normal_channel = normal_channel
		self.npoints = npoints
		self.use_partial_pc = use_partial_pc
		self.is_train = is_train
		self.overfit = args.overfit
		self.one_pose = False
		self.vary_scale = False
		self.more_pose = args.data_more_pose
		self.vary_scale_more_pose = args.data_vary_scale_more_pose
		self.restrict_object_cat = args.restrict_object_cat
		self.cp_s2a_soft = cp_s2a_soft
		self.cp_s2b_soft = cp_s2b_soft

		if self.cp_s2b_soft:
			assert self.cp_s2a_soft

		print('USING PARTIAL PC', use_partial_pc)
		print('VARY SCALE', self.vary_scale)
		print('MORE POSE', self.more_pose)

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

		if self.cp_s2a_soft:
			self.cp_result_folder_dir = os.path.join(self.home_dir_data, 'dataset_cp')
			self.cp_result_more_folder_dir = os.path.join(self.home_dir_data, 'dataset_cp_more')

		if self.cp_s2b_soft:
			exclude_nan_dir = os.path.join(self.home_dir_data, 'exclude', 'cp_s2a_nan.txt')
			self.exclude_nan = load_txt(exclude_nan_dir)

		if use_partial_pc:
			self.pose_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result')
			self.pose_result_more_folder_dir = os.path.join(self.home_dir_data, 'collection_result_more')
			self.pose_result_more_takeoff_dict_dir = os.path.join(self.home_dir_data, 'collection_result_more_seq', 'labels', 'all_dict.json')

			self.pose_result_vary_scale_folder_dir = os.path.join(self.home_dir_data, 'collection_result_vary_scale')
			self.pose_result_vary_scale_more_folder_dir = os.path.join(self.home_dir_data, 'collection_result_vary_scale_more')
			data_list_vary_scale_dir = os.path.join(self.pose_result_vary_scale_folder_dir, 'labels', 'all_list.txt')
			data_list_more_pose_dir = os.path.join(self.pose_result_more_folder_dir, 'labels', 'all_list.txt')
			data_list_vary_scale_more_pose_dir = os.path.join(self.pose_result_vary_scale_more_folder_dir, 'labels', 'all_list.txt')

		else:
			self.cp_result_folder_dir = os.path.join(self.home_dir_data, 'dataset_cp')
			self.pose_result_folder_dir = os.path.join(self.home_dir_data, 'collection_result_old')

		self.all_data = []
		self.all_data_dict = {}
		self.object_cat_dict = {}
		self.data_ct = 0

		# np.random.shuffle(reader)
		reader = open(data_list_dir, 'r').read().splitlines()
		self.load_from_reader(reader, 'ori')
		if self.vary_scale and self.is_train:
			reader_vary_scale = open(data_list_vary_scale_dir, 'r').read().splitlines()
			self.load_from_reader(reader_vary_scale, 'vary_scale')
		if self.more_pose:
			more_pose_filter_dict = load_json(self.pose_result_more_takeoff_dict_dir)
			reader_more_pose = open(data_list_more_pose_dir, 'r').read().splitlines()
			self.load_from_reader(reader_more_pose, 'more_pose', more_pose_filter_dict)
	
		if self.vary_scale_more_pose and self.is_train:
			reader_vary_scale_more_pose = open(data_list_vary_scale_more_pose_dir, 'r').read().splitlines()
			self.load_from_reader(reader_vary_scale_more_pose, 'vary_scale_more_pose')
			
		self.all_data = list(self.all_data_dict.keys())
		self.all_data.sort()
		if self.overfit:
			self.all_data_small = []
			np.random.shuffle(self.all_data)
			for ii in range(len(self.all_data)):
				if len(self.all_data_dict[self.all_data[ii]]) > 3:
					self.all_data_small.append(self.all_data[ii])

					if len(self.all_data_small) == args.batch_size:
						break
			self.all_data = self.all_data_small
			# self.all_data = self.all_data[:1]
			# if not self.is_train:
				# data_include_arr = load_txt('dataset_include.txt')
				# self.all_data = data_include_arr
			print(self.all_data)
		print_object_cat_dict(self.object_cat_dict)

	def load_from_reader(self, reader, dataset_label, filter_dict=None):
		ori = dataset_label == 'ori'
		vary_scale = 'vary_scale' in dataset_label

		for tmp in reader:
			if not vary_scale:
				result_file_name, pose_idx = '_'.join(tmp.split('_')[:-1]), tmp.split('_')[-1]
			else:
				result_file_name_v, pose_idx = split_last(tmp)
				result_file_name, v_n = split_last(result_file_name_v)
			
			# quickfix
			if self.cp_s2b_soft:
				if result_file_name in self.exclude_nan:
					continue
			if not(filter_dict is None):
				if not (tmp in filter_dict):
					continue
				if not filter_dict[tmp]:
					continue
					 
			if (not ori) and (not result_file_name in self.all_data_dict):
				continue 
			hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
			hook_name = '{}_{}'.format(hook_cat, str(hook_id))
			object_name = '{}_{}'.format(object_cat, str(object_id))
			if (not self.restrict_object_cat is None) and (not self.restrict_object_cat == '') and object_cat != self.restrict_object_cat:
				continue
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
			tmp_data_dict = {
				'object_idx': object_idx,
				'hook_idx': hook_idx,
				'pose_idx': int(pose_idx),
				'dataset_label': dataset_label,
			}

			if self.cp_s2a_soft:
				cp_result_file_name = result_file_name + '_' + str(pose_idx)
				
				if dataset_label == 'more_pose':
					cp_map_dir_half = os.path.join(self.cp_result_more_folder_dir, cp_result_file_name)
				else:
					cp_map_dir_half = os.path.join(self.cp_result_folder_dir, cp_result_file_name)
				cp_map_o_dir = cp_map_dir_half + '_cp_map_object.npy'
				cp_map_h_dir = cp_map_dir_half + '_cp_map_hook.npy'
				cp_map_per_o_dir = cp_map_dir_half + '_cp_map_per_object.npy'
				cp_map_per_h_dir = cp_map_dir_half + '_cp_map_per_hook.npy'
				
				if (not os.path.exists(cp_map_o_dir)) or (not os.path.exists(cp_map_h_dir)):
					continue
				
				if self.cp_s2b_soft:
					# assert os.path.exists(cp_map_per_o_dir), cp_map_per_o_dir
					# assert os.path.exists(cp_map_per_h_dir)
					if (not os.path.exists(cp_map_per_o_dir) or (not os.path.exists(cp_map_per_h_dir))):
						continue

				tmp_data_dict['cp_map_o_dir'] = cp_map_o_dir
				tmp_data_dict['cp_map_h_dir'] = cp_map_h_dir

				if self.cp_s2b_soft:
					tmp_data_dict['cp_map_per_o_dir'] = cp_map_per_o_dir
					tmp_data_dict['cp_map_per_h_dir'] = cp_map_per_h_dir

			if not object_cat in self.object_cat_dict:
				self.object_cat_dict[object_cat] = [result_file_name_v if vary_scale else result_file_name] 
			else:
				self.object_cat_dict[object_cat].append(result_file_name_v if vary_scale else result_file_name)

			if not vary_scale:
				data_dict_key = result_file_name
			else:
				data_dict_key = result_file_name_v

			if not (data_dict_key in self.all_data_dict):
				self.all_data_dict[data_dict_key] = [tmp_data_dict]
			else:
				if self.is_train and self.one_pose:
					continue
				self.all_data_dict[data_dict_key].append(tmp_data_dict)
			self.data_ct += 1

	def __getitem__(self, index):
		result_file_name = self.all_data[index]
		data_dict_arr = self.all_data_dict[result_file_name]
		data_dict = data_dict_arr[0]

		vary_scale = 'vary_scale' in data_dict['dataset_label']

		if vary_scale:
			result_file_name_v = result_file_name
			result_file_name, v_n = split_last(result_file_name_v)

		if not self.use_partial_pc:
			point_set_o = np.load(self.all_object_np_dict[data_dict['object_idx']])
		else:
			point_set_o = np.load(self.partial_pc_dir[result_file_name]['object'])
			point_set_o_idx = np.load(self.partial_pc_dir[result_file_name]['object_idx'])

		point_set_o = point_set_o[0:self.npoints,:]
		if not self.normal_channel:
			point_set_o = point_set_o[:,0:3]

		# load hook pc
		if not self.use_partial_pc:
			point_set_h = np.load(self.all_hook_np_dict[data_dict['hook_idx']])
		else:
			point_set_h = np.load(self.partial_pc_dir[result_file_name]['hook'])
			point_set_h_idx = np.load(self.partial_pc_dir[result_file_name]['hook_idx'])

		point_set_h = point_set_h[0:self.npoints,:]
		if not self.normal_channel:
			point_set_h = point_set_h[:,0:3]

		all_pose_o = np.zeros((len(data_dict_arr), 6))

		if vary_scale:
			pose_result_dir = os.path.join(self.pose_result_vary_scale_folder_dir, result_file_name + '.txt')
		else:
			pose_result_dir = os.path.join(self.pose_result_folder_dir, result_file_name + '.txt')
		

		pose_result = load_result_file(pose_result_dir)
		if self.more_pose:
			pose_result_more_dir = os.path.join(self.pose_result_more_folder_dir, result_file_name + '.txt')
			pose_result_more_pose = None
		if self.vary_scale_more_pose:
			pose_result_vary_scale_more_dir = os.path.join(self.pose_result_vary_scale_more_folder_dir, result_file_name + '.txt')
			pose_result_vary_scale_more_pose = None
		
		all_cp_map_o_dir = []
		all_cp_map_h_dir = []
		all_cp_map_per_o_dir = []
		all_cp_map_per_h_dir = []
		for ii, data_dict in enumerate(data_dict_arr):
			pose_idx = data_dict['pose_idx']

			# # load cp map
			# if self.cp_s2a_soft:
			# 	cp_result_file_name = result_file_name + '_' + str(pose_idx)
			# 	if data_dict['dataset_label'] == 'more_pose':
			# 		cp_map_o_dir = os.path.join(self.cp_result_more_folder_dir, cp_result_file_name +  '_cp_map_object.npy')
			# 		cp_map_h_dir = os.path.join(self.cp_result_more_folder_dir, cp_result_file_name +  '_cp_map_hook.npy')
			# 	else:
			# 		cp_map_o_dir = os.path.join(self.cp_result_folder_dir, cp_result_file_name +  '_cp_map_object.npy')
			# 		cp_map_h_dir = os.path.join(self.cp_result_folder_dir, cp_result_file_name +  '_cp_map_hook.npy')

			# 	cp_map_o_ori = np.load(cp_map_o_dir)
			# 	cp_map_h_ori = np.load(cp_map_h_dir)
				
			# 	cp_map_o = cp_map_o_ori[point_set_o_idx]
			# 	cp_map_h = cp_map_h_ori[point_set_h_idx]

			# load pose
			if data_dict['dataset_label'] == 'more_pose':
				if pose_result_more_pose is None:
					pose_result_more_pose = load_result_file(pose_result_more_dir)
				pose_o_end = pose_result_more_pose[pose_idx, -7:]
			elif data_dict['dataset_label'] == 'vary_scale_more_pose':
				if pose_result_vary_scale_more_pose is None:
					pose_result_vary_scale_more_pose = load_result_file(pose_result_vary_scale_more_dir)
				pose_o_end = pose_result_vary_scale_more_pose[pose_idx, -7:]
			else:
				pose_o_end = pose_result[pose_idx,-7:]
			
			pose_quaternion = pose_o_end[3:]
			pose_aa = angle_axis_from_quaternion(pose_quaternion)
			pose_transl_aa = np.zeros((6,))
			pose_transl_aa[:3] = pose_o_end[:3]
			pose_transl_aa[3:] = pose_aa
			pose_np = pose_transl_aa
			all_pose_o[ii] = pose_np
			
			if self.cp_s2a_soft:
				all_cp_map_o_dir.append(data_dict['cp_map_o_dir'])
				all_cp_map_h_dir.append(data_dict['cp_map_h_dir'])

				if self.cp_s2b_soft:
					all_cp_map_per_o_dir.append(data_dict['cp_map_per_o_dir'])
					all_cp_map_per_h_dir.append(data_dict['cp_map_per_h_dir'])
			# plot_pc(point_set_h)
			# plot_pc(point_set_h[cp_map_h], color=[1, 0, 0])
			# from mayavi import mlab as mayalab 
			# mayalab.show()
		
		if vary_scale:
			if data_dict['dataset_label'] == 'vary_scale_more_pose':
				object_scale_abs = np.array(pose_result_vary_scale_more_pose[data_dict['pose_idx'], 1:4])
				object_scale_rel = np.array(pose_result_vary_scale_more_pose[data_dict['pose_idx'], 4:7])
			else:
				object_scale_abs = np.array(pose_result[data_dict['pose_idx'], 1:4])
				object_scale_rel = np.array(pose_result[data_dict['pose_idx'], 4:7])
			point_set_o[:, :3] *= object_scale_rel[np.newaxis, :]

		# for i in range(all_pose_o.shape[0]):
		# 	plot_pc(point_set_h)
		# 	from scipy.spatial.transform import Rotation 
		# 	point_set_o_tmp = np.dot(point_set_o[:, :3], Rotation.from_quat(quaternion_from_angle_axis(all_pose_o[i, 3:])).as_matrix().T) + all_pose_o[i, :3]
		# 	plot_pc(point_set_o_tmp, color=[1, 0, 0])
		# 	from mayavi import mlab as mayalab 
		# 	mayalab.show()

		# 	if self.cp_s2a_soft:
		# 		plot_pc_s(point_set_o, cp_map_o)
		# 		mayalab.show()
		# 		plot_pc_s(point_set_h, cp_map_h)
		# 		mayalab.show()
				

		return {
			'pc_o':point_set_o,
			'pc_h':point_set_h, 
			'pc_o_idx': point_set_o_idx,
			'pc_h_idx': point_set_h_idx,
			# 'cp_map_o':cp_map_o if self.cp_s2a_soft else None,
			# 'cp_map_h':cp_map_h if self.cp_s2a_soft else None,
			'cp_map_o': None,
			'cp_map_h': None,
			'cp_map_o_dir': all_cp_map_o_dir,
			'cp_map_h_dir': all_cp_map_h_dir,
			'cp_map_per_o_dir': all_cp_map_per_o_dir,
			'cp_map_per_h_dir': all_cp_map_per_h_dir,
			'pose_o':all_pose_o,
			'n_pose': all_pose_o.shape[0],
			'result_file_name': result_file_name_v if vary_scale else result_file_name,
		}

	def __len__(self):
		return len(self.all_data)

	@staticmethod
	def pad_collate_fn_for_dict(batch):
		pc_o_batch = [d['pc_o'] for d in batch]
		pc_o_batch = np.stack(pc_o_batch, axis=0)

		pc_h_batch = [d['pc_h'] for d in batch]
		pc_h_batch = np.stack(pc_h_batch, axis=0)

		pc_o_idx_batch = [d['pc_o_idx'] for d in batch]
		pc_h_idx_batch = [d['pc_h_idx'] for d in batch]

		cp_map_o_batch = [d['cp_map_o'] for d in batch]
		cp_map_h_batch = [d['cp_map_h'] for d in batch]

		cp_map_o_dir_batch = [d['cp_map_o_dir'] for d in batch]
		cp_map_h_dir_batch = [d['cp_map_h_dir'] for d in batch]
		cp_map_per_o_dir_batch = [d['cp_map_per_o_dir'] for d in batch]
		cp_map_per_h_dir_batch = [d['cp_map_per_h_dir'] for d in batch]

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
			'output1': cp_map_o_batch, # (batch_size, [varies], 2) list of np.int64 array
			'output2': cp_map_h_batch, # (batch_size, [varies], 2) list of np.int64 array
			'output3': None, # (batch_size, 4096, 4096) np.bool_
			'output4': pose_o_batch, # (batch_size, 7) np.float64
			'result_file_name': result_file_name,
			'n_pose': n_pose,
			'pc_o_idx': pc_o_idx_batch,
			'pc_h_idx': pc_h_idx_batch,
			'cp_map_o_dir': cp_map_o_dir_batch,
			'cp_map_h_dir': cp_map_h_dir_batch,
			'cp_map_per_o_dir': cp_map_per_o_dir_batch,
			'cp_map_per_h_dir': cp_map_per_h_dir_batch,
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
	import argparse
	from torch.utils.data import DataLoader
	import torch
	torch.manual_seed(2)
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument('--bohg4', action='store_true')

	parser.add_argument('--pointset_dir', default='/scr2/')
	parser.add_argument('--restrict_object_cat', default='')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--data_one_pose', action='store_true')
	parser.add_argument('--data_vary_scale', action='store_true')
	parser.add_argument('--data_more_pose', action='store_true')
	parser.add_argument('--data_vary_scale_more_pose', action='store_true')
	parser.add_argument('--cp', action='store_true')

	args = parser.parse_args()
	if args.bohg4:
		args.home_dir_data = '/scr1/yifan/hang'

	home_dir_data = args.home_dir_data
	cp_result_folder_dir= os.path.join(home_dir_data,'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir,'labels','train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir,'labels','test_list.txt')

	train_set = MyDataset(home_dir_data, train_list_dir, True, use_partial_pc=True, cp_s2a_soft=args.cp, cp_s2b_soft=args.cp, args=args)
	test_set = MyDataset(home_dir_data, test_list_dir, False, use_partial_pc=True, cp_s2a_soft=args.cp, cp_s2b_soft=args.cp, args=args)

	print('len train', len(train_set))
	print('len test',len(test_set))
	train_loader = DataLoader(train_set, batch_size=2, shuffle=True,
					num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	test_loader = DataLoader(test_set, batch_size=2, shuffle=True,
					num_workers=1, collate_fn=MyDataset.pad_collate_fn_for_dict)
	import time
	t_a = time.time()
	for i, one_data in enumerate(test_loader):
		# print(i)
		print(len(train_loader))
		# print('pc o', one_data['input1'].shape)
		# print('pc h', one_data['input2'].shape)
		# print('pose o end', one_data['output4'].shape)
		# # print('pose o end', one_data['output4'])
		# print('cp_map_o', len(one_data['output2']))
		# print('cp_map_h', len(one_data['output1']))
		print(one_data['n_pose'])
		if i > 10:
			break
	print(time.time() - t_a)
	pass
