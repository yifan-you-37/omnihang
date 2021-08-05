import numpy as np
import os
import sys
import pickle
import glob
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)

from data_helper import *
from coord_helper import *
from rotation_lib import *

class ReplayBuffer(object):
	def __init__(self, npoints, action_dim, dataset, save_dir, points_dim=3, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.dataset = dataset

		self.npoints = npoints
		# self.state = np.zeros((max_size, npoints * 2 + 2048, points_dim+1))
		self.pose = np.zeros((max_size, 6))
		self.reward = np.zeros((max_size,))
		self.hook_name = np.array(['abc'] * max_size, dtype = 'object')
		self.object_name = np.array(['abc'] * max_size, dtype = 'object')
		self.preload_result_dir = np.array(['a'] * max_size, dtype = 'object')
		self.preload_pose_idx = np.array(['a'] * max_size, dtype = 'object')
		self.if_preload = np.zeros((max_size), dtype=np.bool_)

		self.new_pose_dict = {}
		self.save_dir = save_dir

	def add(self, one_object_name, one_hook_name, one_pose, one_reward, resave=True):
		result_file_name = '{}_{}'.format(one_hook_name, one_object_name)
		self.object_name[self.ptr] = one_object_name
		self.hook_name[self.ptr] = one_hook_name
		self.pose[self.ptr] = one_pose
		self.reward[self.ptr] = one_reward

		if resave:
			if result_file_name in self.new_pose_dict:
				self.new_pose_dict[result_file_name].append(one_pose)
			else:
				self.new_pose_dict[result_file_name] = [one_pose]

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	def add_bulk(self, object_name, hook_name, pose, reward):
		n_add = pose.shape[0]
		assert n_add == reward.shape[0]
		self.object_name[self.ptr:self.ptr+n_add] = object_name
		self.hook_name[self.ptr:self.ptr+n_add] = hook_name
		self.pose[self.ptr:self.ptr+n_add] = pose
		self.reward[self.ptr:self.ptr+n_add] = reward

		self.ptr = (self.ptr + n_add) % self.max_size
		self.size = min(self.size + n_add, self.max_size)


	def add_preload(self, one_object_name, one_hook_name, one_result_dir, one_pose_idx, one_reward):
		self.if_preload[self.ptr] = True
		self.hook_name[self.ptr] = one_hook_name
		self.object_name[self.ptr] = one_object_name
		self.preload_result_dir[self.ptr] = one_result_dir
		self.preload_pose_idx[self.ptr] = one_pose_idx
		self.reward[self.ptr] = one_reward
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)		

	def save_txt(self):
		for result_file_name, poses in self.new_pose_dict.items():
			out_dir = os.path.join(self.save_dir, result_file_name + '.txt')
			with open(out_dir, 'a+') as f:
				for pose in poses:
					f.write(comma_separated(pose) + '\n')
		self.new_pose_dict = {}

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		pc_combined = np.zeros((batch_size, self.npoints * 2, 4))
		for i in range(batch_size):
			idx = ind[i]
			o_name = self.object_name[idx]
			h_name = self.hook_name[idx]
			if not self.if_preload[idx]:
				cur_pose = self.pose[idx]
			else:
				cur_pose = load_result_file(self.preload_result_dir[idx])[self.preload_pose_idx[idx], -7:]

			result_file_name = h_name + '_' + o_name

			pc_o = np.load(self.dataset.partial_pc_dir[result_file_name]['object'])[:, :3]
			pc_h = np.load(self.dataset.partial_pc_dir[result_file_name]['hook'])[:, :3]
			
			cur_pc_combined = create_pc_combined(pc_o, pc_h, cur_pose[:3], cur_pose[3:])
			pc_combined[i] = cur_pc_combined
		return (
			pc_combined,
			self.reward[ind],
		)

class ReplayBufferTwo(object):
	def __init__(self, npoints, action_dim, dataset, home_dir_data, points_dim=3, max_size=int(1e6)):
		self.dataset = dataset
		succ_save_dir = os.path.join(home_dir_data, 'collection_result_s3_pos')
		fail_save_dir = os.path.join(home_dir_data, 'collection_result_s3_neg')
		# if not os.path.exists(succ_save_dir):
		# 	os.mkdir(succ_save_dir)
		# if not os.path.exists(fail_save_dir):
		# 	os.mkdir(fail_save_dir)
		self.buffer_fail = ReplayBuffer(npoints, action_dim, dataset, fail_save_dir, points_dim, max_size)
		self.buffer_succ = ReplayBuffer(npoints, action_dim, dataset, succ_save_dir, points_dim, max_size)

	def add(self, object_name, hook_name, pose, reward):
		if reward == 1:
			self.buffer_succ.add(object_name, hook_name, pose, reward)
		else:
			self.buffer_fail.add(object_name, hook_name, pose, reward)

	def sample(self, batch_size):
		if self.buffer_succ.size != 0:
			batch_succ = self.buffer_succ.sample(int(batch_size/2))
			batch_fail = self.buffer_fail.sample(batch_size-int(batch_size/2))

			return (
				np.append(batch_succ[0], batch_fail[0], axis=0),
				np.append(batch_succ[1], batch_fail[1], axis=0),
			)
		else:
			return self.buffer_fail.sample(batch_size)
	
	def save_txt(self):
		print('saving replay buffer...')
		self.buffer_succ.save_txt()
		self.buffer_fail.save_txt()

	def preload_data(self):
		print('preloading data...')
		# tmp_hook_name = 'hook_wall_163'
		# tmp_object_name = 'mug_127'
		for data_dict in self.dataset.all_data:
			# if data_dict['object_name'] != tmp_object_name or data_dict['hook_name'] != tmp_hook_name:
				# continue 
			if data_dict['label'] == 1:
				self.buffer_succ.add_preload(data_dict['object_name'], data_dict['hook_name'], data_dict['result_dir'], data_dict['pose_idx'], data_dict['label'])
			else:
				self.buffer_fail.add_preload(data_dict['object_name'], data_dict['hook_name'], data_dict['result_dir'], data_dict['pose_idx'], data_dict['label'])
		print('preloaded n success {} n fail {}'.format(self.buffer_succ.size, self.buffer_fail.size))

	def load_txt(self, succ_save_dir, fail_save_dir):
		for result_file in os.listdir(succ_save_dir):
			result_file_name = result_file[:-4]
			hook_name, object_name = split_result_file_name(result_file_name)
			result_np = load_result_file(os.path.join(succ_save_dir, result_file))
			self.buffer_succ.add_bulk(object_name, hook_name, result_np, np.ones((result_np.shape[0])))
		for result_file in os.listdir(fail_save_dir):
			result_file_name = result_file[:-4]
			hook_name, object_name = split_result_file_name(result_file_name)
			result_np = load_result_file(os.path.join(fail_save_dir, result_file))
			print(os.path.join(fail_save_dir, result_file), result_np.shape[0])
			self.buffer_fail.add_bulk(object_name, hook_name, result_np, np.ones((result_np.shape[0])))

	@staticmethod
	def save_pkl(out_folder_dir, buffer):
		if not os.path.isdir(out_folder_dir):
			os.mkdir(out_folder_dir)

		out_dir = os.path.join(out_folder_dir, '{}.pkl'.format(datetime.datetime.now().strftime('%b%d_%H-%M-%S')))
		# assert not os.path.exists(out_dir)
		pickle.dump(buffer, open(out_dir, 'wb'))

def combine_replay_buffer(buffer_name_list, dataset, home_dir_data, preload_data=False):
	print('combining replay buffer...', buffer_name_list)
	buffer_list = []
	for buffer_name in buffer_name_list:
		if buffer_name is None:
			continue
		buffer_list.append(pickle.load(open(buffer_name, 'rb')))

	my_buffer = ReplayBufferTwo(4096, 7, dataset, home_dir_data, max_size=int(1e6))
	if preload_data:
		my_buffer.preload_data()
	for buffer in buffer_list:
		# do not load the pre-loaded data

		# succ buffer
		n_preload = np.sum(buffer.buffer_succ.if_preload)
		assert np.sum(buffer.buffer_succ.if_preload[:n_preload]) == n_preload
		n_total = buffer.buffer_succ.size

		pose_add = buffer.buffer_succ.pose[n_preload:n_total]
		reward_add = buffer.buffer_succ.reward[n_preload:n_total]

		object_name_add = buffer.buffer_succ.object_name[n_preload:n_total]
		hook_name_add = buffer.buffer_succ.hook_name[n_preload:n_total]
		my_buffer.buffer_succ.add_bulk(object_name_add, hook_name_add, pose_add, reward_add)

		# fail buffer
		n_preload = np.sum(buffer.buffer_fail.if_preload)
		assert np.sum(buffer.buffer_fail.if_preload[:n_preload]) == n_preload
		n_total = buffer.buffer_fail.size

		pose_add = buffer.buffer_fail.pose[n_preload:n_total]
		reward_add = buffer.buffer_fail.reward[n_preload:n_total]

		object_name_add = buffer.buffer_fail.object_name[n_preload:n_total]
		hook_name_add = buffer.buffer_fail.hook_name[n_preload:n_total]

		my_buffer.buffer_fail.add_bulk(object_name_add, hook_name_add, pose_add, reward_add)		
	print('replay buffer', my_buffer.buffer_succ.size, my_buffer.buffer_fail.size)

	return my_buffer

if __name__ == '__main__':
	from classifier_dataset_torch import ClassifierDataset
	home_dir_data = '/scr1/yifan/hang'
	
	cp_result_folder_dir = os.path.join(home_dir_data, 'dataset_cp')
	
	train_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'train_list.txt')

	train_set = ClassifierDataset(home_dir_data, train_list_dir, False, split='train')
	replay_buffer = ReplayBufferTwo(4096, 7, train_set, home_dir_data, max_size=int(1e6))
	replay_buffer.preload_data()

	replay_buffer.load_txt('/scr1/yifan/hang/collection_result_s3_pos', '/scr1/yifan/hang/collection_result_s3_neg')
	print(replay_buffer.buffer_succ.size, replay_buffer.buffer_fail.size)
	import time
	a = time.time()

	# for i in range(100):
		# replay_buffer.sample(100)
		# print(i)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 1)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 1)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 1)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 0)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 0)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 0)
	# replay_buffer.add('hook_wall_1', 'cap_2', np.zeros((7)), 0)
	# ReplayBufferTwo.save_pkl('test_pkl', replay_buffer)
	# time.sleep(1)
	# ReplayBufferTwo.save_pkl('test_pkl', replay_buffer)
	# time.sleep(1)
	# ReplayBufferTwo.save_pkl('test1_pkl', replay_buffer)
	# time.sleep(1)
	# ReplayBufferTwo.save_pkl('test1_pkl', replay_buffer)
	# # print(replay_buffer.buffer_succ.size, replay_buffer.buffer_succ.if_preload[:10])
	# # out_file = get_2nd_last_dir('../saved_models/pose_cls_no_wall', '*model.ckpt.index')
	# # print(out_file)
	print(time.time() - a)

	# my_buffer = pickle.load(open(get_2nd_last_dir('test1_pkl'), 'rb'))
	# print(my_buffer.buffer_succ.size, my_buffer.buffer_succ.if_preload[:10])
	buffer_name_list = [get_2nd_last_dir('test1_pkl'), get_2nd_last_dir('test_pkl')]
	my_buffer = combine_replay_buffer(buffer_name_list, train_set, home_dir_data)
	print(my_buffer.buffer_succ.size)
	print(my_buffer.buffer_fail.size)
