import time
import numpy as np
import random
import sys
import os
import argparse
# import cv2
import zipfile
import itertools
import pybullet
import json
import time
import numpy as np
import imageio

from collect_pose_data import PoseDataCollector
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file
import bullet_client as bc

class SeqDataCollector(PoseDataCollector):
	def __init__(self, p_id):
		super(SeqDataCollector, self).__init__(p_id)
		self.n_f = 1
		self.f_interval = 40
		self.f_mag = 20.
		self.pic_freq = 5
	def try_one_plan(self, f_arr, hook_bb, object_bullet_id, object_world_pos):
		# penetration = False
		succ = False
		fail = False
		# print(hook_bb)
		# self.p.addUserDebugLine([0, 0, 0], object_world_pos, lineWidth=18)
		# drawAABB(hook_bb, self.p)
		
		object_pos_quat_seq = []
		image_seq = []
		
		for i in range(self.n_f):
			f = f_arr[i]
			cur_object_quat = self.p.getBasePositionAndOrientation(object_bullet_id)[1]
			_, inv_quat = self.p.invertTransform([0, 0, 0], cur_object_quat)
			rotmat = np.array(self.p.getMatrixFromQuaternion(inv_quat)).reshape(3, 3)
			f_local = np.matmul(rotmat, f)


			for t in range(self.f_interval):
				self.p.applyExternalForce(object_bullet_id, -1, f_local, [0, 0, 0], flags=self.p.LINK_FRAME)

				cur_object_pos, cur_object_quat = self.p.getBasePositionAndOrientation(object_bullet_id)
				object_pos_quat_seq.append(cur_object_pos + cur_object_quat)
				# if t % self.pic_freq == 0:
					# image_seq.append(self.env.take_pic())
				self.p.stepSimulation()
				# time.sleep(0.1)
				if t % 10 == 0:
					# cur_object_pos, cur_object_quat = self.p.getBasePositionAndOrientation(object_bullet_id)
					# cur_rotmat = np.array(self.p.getMatrixFromQuaternion(cur_object_quat)).reshape(3, 3)
					# f_global = np.matmul(cur_rotmat, f_local) / self.f_mag
					# self.p.addUserDebugLine(cur_object_pos, cur_object_pos + f_global)
					object_bb = np.array(self.p.getAABB(object_bullet_id, -1))
					# drawAABB(object_bb, self.p)
					# for tmp in self.p.getContactPoints(hook_bullet_id, object_bullet_id):
						# if tmp[8] < -0.001:
							# penetration = True
							# break
					# if penetration:
						# fail = True
						# break
					if not is_overlapping_3d(hook_bb, object_bb):
						succ = True
						break
		# if not succ:
		for t in range(20):
			cur_object_pos, cur_object_quat = self.p.getBasePositionAndOrientation(object_bullet_id)
			if not succ:
				object_pos_quat_seq.append(cur_object_pos + cur_object_quat)
			# if t % self.pic_freq == 0:
				# image_seq.append(self.env.take_pic())
			self.p.stepSimulation()

		object_bb = np.array(self.p.getAABB(object_bullet_id, -1))
		if (not succ) and (not is_overlapping_3d(hook_bb, object_bb)):
			succ = True
		
		if succ:
			# image_seq.append(self.env.take_pic())
			return True, object_pos_quat_seq, None

		return False, None, None
		

	def take_off_one_pair(self, hook_bullet_id, object_bullet_id, hook_scaling, object_scaling, object_init_pos_world, object_init_quat_world, result_folder_dir, result_file_name, mass_fric_dict):
		# if os.path.isfile(output_json_dir):
			# print('skip')
			# return True
		hook_bb = np.array(self.p.getAABB(hook_bullet_id, 0)) 
	
		# all_forces = np.array([ [0, 1, 0],[-1, 0, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0],]) # [0, 0, -1]
		all_forces = np.array([ [-1, -1, 1], [-1, 1, 1], [1, 0, 1], [0, 1, 1],[0, -1, 1], [-1, 0, 1], [0, 1, 0],[-1, 0, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0]]) # [0, 0, -1]
	
		all_forces = 1. * all_forces / np.linalg.norm(all_forces, axis=1, keepdims=True)
		succ = False

		succ_force_all = []
		ct = 0
		for i in range(all_forces.shape[0]):
			# for j in range(all_forces.shape[0]):
				j = 0
				# print(i, j)
				self.p.resetBasePositionAndOrientation(object_bullet_id, object_init_pos_world, object_init_quat_world)
				if self.check_object_touches_ground(object_bullet_id):
					break

				f_arr = all_forces[[i, j]] * self.f_mag
				flag, object_pos_quat_seq, _ = self.try_one_plan(f_arr, hook_bb, object_bullet_id, object_init_pos_world)

				if not flag:
					continue

				succ = True

				# convert to hook frame
				object_pos_quat_seq = np.array(object_pos_quat_seq)
				object_pos_quat_seq[:, :3] = global_pos_to_local(object_pos_quat_seq[:, :3], hook_world_pos, 1.)
				succ_force_all.append((f_arr[:self.n_f]).tolist())
				
				half_output_dir = os.path.join(result_folder_dir, result_file_name + '-{}'.format(int(ct)))
				# gif_output_dir = half_output_dir + '.gif'
				np_output_dir = half_output_dir + '-pose.npy'

				print(object_pos_quat_seq.shape)
				np.save(np_output_dir, object_pos_quat_seq)
				# imageio.mimsave(gif_output_dir, image_seq, fps=5)

				print('force', i, 'success')
				ct += 1
				break

		assert ct == len(succ_force_all)
		# if ct > 0:
		output_dict = {
			'succ_force': succ_force_all
		}
		# with open(output_json_dir, 'w+') as f:
			# json.dump(output_dict, f)
			
		if succ:
			return True, output_dict
		return False, output_dict	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--sherlock", action='store_true')
	parser.add_argument("--hook_name", default='')
	parser.add_argument('--object_category', default='')
	parser.add_argument('--object_name', default='')
	args = parser.parse_args()

	if args.sherlock:
		args.home_dir_data = '/scratch/groups/bohg/hang'
		assert args.hook_name != ''
		assert args.object_category != ''
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_more')
	seq_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_more_seq')

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_category(data_dir, args.object_category)

	p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
	collector = SeqDataCollector(p_id) 

	if not os.path.isdir(seq_result_folder_dir):
		os.mkdir(seq_result_folder_dir)
	mkdir_if_not(os.path.join(seq_result_folder_dir, 'labels'))

	ct = 0

	if args.hook_name != '':
		assert args.hook_name in all_hook_name

	assert args.hook_name != ''

	result_labels_dict = {}
	if args.object_name == '':
		out_labels_dir = os.path.join(seq_result_folder_dir, 'labels', '{}_{}.json'.format(args.hook_name, args.object_category))
	else:
		out_labels_dir = os.path.join(seq_result_folder_dir, 'labels', '{}_{}_{}.json'.format(args.hook_name, args.object_category, args.object_name))

	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		hook_urdf_dir = all_hook_urdf[i]
		hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf_dir)
		hook_pc = np.load(hook_pc_dir)
		hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)
		hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
		hook_world_pos = collector.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
		hook_bb = collector.p.getAABB(hook_bullet_id, 0)
		make_wall(hook_bb, hook_world_pos, collector.p, wall_size=0.5)
		# print('hook world pos offest', hook_world_pos_offset)
		
		for j, object_name in enumerate(all_object_name):
			if args.object_name != '' and args.object_name != object_name:
				continue
			object_urdf_dir = all_object_urdf[j]
			object_pc_dir = get_numpy_dir_from_urdf(object_urdf_dir)
			object_pc = np.load(object_pc_dir)
			result_file_name = hook_name + '_' + object_name
			result_file_dir = os.path.join(collection_result_folder_dir, result_file_name + '.txt')
			if not os.path.isfile(result_file_dir):
				continue
			result_np = load_result_file(result_file_dir)
			included_rows = list(range(result_np.shape[0]))

			ct += 1
			object_bullet_id = collector.p.loadURDF(object_urdf_dir, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)
			object_scaling = collector.p.getCollisionShapeData(object_bullet_id, -1)[0][3][0] 	

			info_output_file_dir = os.path.join(seq_result_folder_dir, '{}.json'.format(result_file_name))
			info_output_arr = []


			object_mass, hook_mass = np.random.uniform(0.3, 0.8, size=[2])
			object_fric, hook_fric = np.random.uniform(0.1, 0.5, size=[2])
			collector.p.changeDynamics(object_bullet_id, -1, lateralFriction=object_fric, mass=object_mass)
			collector.p.changeDynamics(hook_bullet_id, 0, lateralFriction=hook_fric, mass=hook_mass)
			
			mass_fric_dict = {
				'object_fric': object_fric,
				'hook_fric': hook_fric,
				'object_mass': object_mass,
				'hook_mass': hook_mass
			}
			all_output_dict = {}
			all_output_dict['mass_fric'] = mass_fric_dict
			all_output_dict['forces'] = []
			result_labels_dict[result_file_name] = {}

			for k in range(result_np.shape[0]):
				if not (k in included_rows):
					continue
				object_pos_local = result_np[k, -7:-4]
				object_quat_local = result_np[k, -4:]
				object_pos_world = local_pos_to_global(object_pos_local, hook_world_pos, 1.)
				print(k, result_file_name)
				flag_tmp, output_dict = collector.take_off_one_pair(hook_bullet_id, object_bullet_id, hook_scaling, object_scaling, object_pos_world, object_quat_local, \
							result_folder_dir=seq_result_folder_dir, result_file_name='{}-{}'.format(result_file_name, str(k)), \
							mass_fric_dict=mass_fric_dict)
				all_output_dict['forces'].append(output_dict)
				result_labels_dict[result_file_name][k] = flag_tmp

			save_json(info_output_file_dir, all_output_dict)

			# break
			collector.p.removeBody(object_bullet_id)
			if (ct + 1) % 10 == 0:
				print('reset')
				collector.p.disconnect()
				p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
				collector = SeqDataCollector(p_id)
				hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)
				make_wall(hook_bb, hook_world_pos, collector.p, wall_size=0.5)
				
		# break
		collector.p.removeBody(hook_bullet_id)
	save_json(out_labels_dir, result_labels_dict)