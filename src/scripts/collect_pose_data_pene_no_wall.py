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

def my_sample_points_in_bb_uniform(n, bb_low, bb_high):
	# n_dim = int(math.ceil(n**(1./3)))
	sample_x = np.random.uniform(bb_low[0], bb_high[0], size=n)
	sample_y = np.random.uniform(bb_low[1], bb_high[1], size=n)
	sample_z = np.random.uniform(bb_low[2], bb_high[2], size=n)

	# return cartesian_product(sample_x, sample_y, sample_z)
	# print(np.stack([sample_x, sample_y, sample_z], axis=1).shape)
	return np.stack([sample_x, sample_y, sample_z], axis=1)

def sample_points_hook_for_collide(hook_bb, bb_extra_dist=0.3, p=None):
	bb_upper = np.max(hook_bb,axis=0)
	bb_lower = np.min(hook_bb,axis=0)
	bb_extra_dist *= 1.3
	bb_extra_dist = min(0.5, bb_extra_dist)
	bb_upper += bb_extra_dist
	# bb_upper[1] += 0.3
	# bb_upper[2] += 0.3
	bb_lower -= bb_extra_dist

	# bb_upper[2] += 0.2
	# p.addUserDebugLine(bb_lower,bb_upper, lineWidth=19)

	# uniform_points = sample_points_in_sphere_uniform(1000, center=(bb_upper + bb_lower) / 2, radius=np.linalg.norm(bb_upper - bb_lower)/2.)
	uniform_points = my_sample_points_in_bb_uniform(int(15**3), bb_lower, bb_upper)
	# drawAABB([np.min(uniform_points, axis=0), np.max(uniform_points, axis=0)], p)
	# input('bb1')
	# drawAABB([bb_lower, bb_upper], p)
	# input('bb2')
	# filter_mask = filter_inside_bb(uniform_points, bb_lower, bb_upper)
	# uniform_points = uniform_points[filter_mask]

	return uniform_points

def dict_to_csv(out_dir, all_data):
	print('writing', out_dir)
	w = csv.writer(open(out_dir, "w+"))
	for key, val in all_data.items():
		w.writerow([key, val])

class PeneDataCollector(PoseDataCollector):
	def __init__(self, p_id):
		super(PeneDataCollector, self).__init__(p_id)
	
	def collect_pene_data_one_hook_object(self, hook_bullet_id, object_bullet_id, hook_scaling, object_scaling, hook_world_pos,
		hook_pc_n, object_pc_n, hook_tree, result_file_poses):
		try:
			hook_bb = self.p.getAABB(hook_bullet_id, 0)
		except:
			hook_bb = self.p.getAABB(hook_bullet_id, -1)
		object_bb = self.p.getAABB(object_bullet_id)
		ox, oy, oz = object_bb[1][0] - object_bb[0][0], object_bb[1][1] - object_bb[0][1], object_bb[1][2] - object_bb[0][2]
		# print('ox', ox, oy, oz)
		potential_pos_world = sample_points_hook_for_collide(hook_bb, np.max([ox, oy, oz]), self.p, )
		potential_quat = sample_quat_uniform(potential_pos_world.shape[0])

		ct = 0
		pos_result_arr = []
		neg_result_arr = []

		for i in range(potential_pos_world.shape[0]):
			# pose = result_file_poses[0][-7:]
			object_pos_local = potential_pos_world[i] - hook_world_pos
			object_quat_local = potential_quat[i]
			# object_pos_local = pose[:3]
			# object_quat_local = pose[3:]
			
			pene = False
			# hook wall 3 bag 2
			# pose = np.array([0.5605147, -0.00370588, 0.9306318, -0.74062765, -0.28224521, 0.26840015, 0.54751227])

			self.p.resetBasePositionAndOrientation(object_bullet_id, potential_pos_world[i], potential_quat[i])
			# self.p.resetBasePositionAndOrientation(object_bullet_id, pose[:3] + hook_world_pos, pose[3:])
			if self.check_object_touches_ground(object_bullet_id):
				continue

			# use getContactPoint	
			# self.p.stepSimulation()

			# for tmp in self.p.getContactPoints(hook_bullet_id, object_bullet_id):
			# 	if tmp[8] < -0.001:
			# 		pene = True
			# 		break
			# object_pos_local, object_quat_local = self.p.getBasePositionAndOrientation(object_bullet_id)
			# object_pos_local -= hook_world_pos


			# use getClosesetPoint
			close_points = self.p.getClosestPoints(hook_bullet_id, object_bullet_id, distance=0.01)
			if len(close_points) == 0:
				continue
			for tmp in close_points:
				if tmp[8] < -0.001:
					pene = True
					break


			if pene: 
				if len(pos_result_arr) < 20:
					pos_result_arr.append([
						hook_scaling, object_scaling, *object_pos_local, *object_quat_local 
					])
			elif not pene:
				if len(neg_result_arr) < 20:
					neg_result_arr.append([
						hook_scaling, object_scaling, *object_pos_local, *object_quat_local 
					])
			
			if len(pos_result_arr) > 5 and len(neg_result_arr) > 5:
				break

			# if ct >= 10:
				# break
		return pos_result_arr, neg_result_arr

from scipy.spatial import KDTree
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
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
	pos_collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_pene_pos')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	neg_collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_pene_neg')
	pos_labels_dir = os.path.join(pos_collection_result_folder_dir, 'labels')
	neg_labels_dir = os.path.join(neg_collection_result_folder_dir, 'labels')

	mkdir_if_not(pos_collection_result_folder_dir)
	mkdir_if_not(neg_collection_result_folder_dir)
	mkdir_if_not(pos_labels_dir)
	mkdir_if_not(neg_labels_dir)

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, labels_folder_dir, True, True, with_wall=False)
	p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)

	if not os.path.exists(neg_collection_result_folder_dir):
		os.mkdir(neg_collection_result_folder_dir)
	collector = PeneDataCollector(p_id)
	ct = 0
	
	tmp_total_ct = 0
	tmp_succ_ct = 0
	tmp_fail_ct = 0
	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		
		out_pos_labels_dir = os.path.join(pos_labels_dir, '{}.txt'.format(hook_name))
		out_neg_labels_dir = os.path.join(neg_labels_dir, '{}.txt'.format(hook_name))
		if os.path.exists(out_pos_labels_dir) and os.path.exists(out_neg_labels_dir):
			print('skip', hook_name)
			continue

		hook_urdf = all_hook_urdf[i]
		hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf)
		hook_world_pos_offset = get_hook_wall_offset(hook_urdf)
		hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf)
		hook_world_pos = collector.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
		hook_pc = np.load(hook_pc_dir)	
		hook_tree = KDTree(hook_pc[:, :3], leafsize=1000)

		num_pos_dict = {}
		num_neg_dict = {}

		for j, object_name in enumerate(all_object_name):
			# if not 'daily_object' in object_name:
				# continue
			object_urdf = all_object_urdf[j]
			object_pc_dir = get_numpy_dir_from_urdf(object_urdf)
			object_pc = np.load(object_pc_dir)

			result_file_name = hook_name + '_' + object_name
			print(result_file_name)
			neg_out_dir = os.path.join(neg_collection_result_folder_dir, result_file_name + '.txt')
			pos_out_dir = os.path.join(pos_collection_result_folder_dir, result_file_name + '.txt')
			result_dir = os.path.join(collection_result_folder_dir, result_file_name+ '.txt')
			if not os.path.isfile(result_dir):
				continue
			result_file_poses = load_result_file(result_dir)
			if result_file_poses.shape[0] == 0:
				continue
			# if os.path.isfile(out_dir):
				# continue
			
			ct += 1
			object_bullet_id = collector.p.loadURDF(object_urdf, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)
			object_scaling = collector.p.getCollisionShapeData(object_bullet_id, -1)[0][3][0] 	

			pos_result_arr, neg_result_arr = collector.collect_pene_data_one_hook_object(hook_bullet_id, object_bullet_id, hook_scaling, object_scaling, hook_world_pos, 
				hook_pc, object_pc, hook_tree, result_file_poses)

			# for kk in range(result_file_poses.shape[0]):
			# 	pose = result_file_poses[kk]
			# 	# print(pose)
			# 	collector.p.resetBasePositionAndOrientation(object_bullet_id, pose[-7:-4] + hook_world_pos, pose[-4:])
			# 	if collector.check_object_touches_ground(object_bullet_id):
			# 		continue
			# 	tmp_total_ct += 1
			# 	new_pc = np.dot(object_pc[:, :3], Rotation.from_quat(pose[-4:]).as_matrix().T) + pose[-7:-4]
			# 	pene = calc_penetration(new_pc, hook_pc, hook_tree)
			# 	# from mayavi import mlab as mayalab 
			# 	# plot_pc(new_pc)
			# 	# plot_pc(hook_pc)
			# 	# mayalab.show()
			# 	# close_points = collector.p.getClosestPoints(hook_bullet_id, object_bullet_id, linkIndexA=0, linkIndexB=-1, distance=0.01)
			# 	# if len(close_points) == 0:
			# 		# continue
			# 	# pene = False
			# 	# for tmp in close_points:
			# 		# if tmp[8] < -0.001:
			# 			# pene = True
			# 			# print('penetration problem')
			# 			# break
			# 	if pene:
			# 		tmp_fail_ct += 1
			# 	else:
			# 		tmp_succ_ct += 1
			# 	print(pene, tmp_succ_ct, tmp_fail_ct, tmp_total_ct)
			# 	break
				# input('rw')
			num_pos_dict[result_file_name] =  len(pos_result_arr)
			num_neg_dict[result_file_name] =  len(neg_result_arr)

			print(len(pos_result_arr), len(neg_result_arr), result_file_name)
			with open(pos_out_dir, 'w+') as f:
				for result in pos_result_arr:
					f.write(comma_separated(result) + '\n')
			with open(neg_out_dir, 'w+') as f:
				for result in neg_result_arr:
					f.write(comma_separated(result) + '\n')
			# print(pos_out_dir, neg_out_dir)
			collector.p.removeBody(object_bullet_id)
			if (ct + 1) % 30 == 0:
				print('reset')
				collector.p.disconnect()
				p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
				collector = PeneDataCollector(p_id)
				hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf)

		out_pos_labels_dir = os.path.join(pos_labels_dir, '{}.txt'.format(hook_name))
		out_neg_labels_dir = os.path.join(neg_labels_dir, '{}.txt'.format(hook_name))
		collector.p.removeBody(hook_bullet_id)	

		dict_to_csv(out_pos_labels_dir, num_pos_dict)
		dict_to_csv(out_neg_labels_dir, num_neg_dict)


	# for j in range(20000):
		# collector.p.stepSimulation()
		# time.sleep(1./240.)