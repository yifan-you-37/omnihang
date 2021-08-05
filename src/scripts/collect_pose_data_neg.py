import pybullet 
import time
import numpy as np
import random
# np.random.seed(5)
# random.seed(5)
import sys
import os
import argparse
# import cv2

from collect_pose_data import PoseDataCollector
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
from collision_helper import *
import bullet_client as bc

def sample_neg_points_hook(hook_bb, object_bb, p=None):
	bb_upper = np.max(hook_bb,axis=0)
	bb_lower = np.min(hook_bb,axis=0)
	# bb_upper -= 0.01
	# bb_lower -= 0.003
	object_xx = object_bb[1][0] - object_bb[0][0]
	object_yy = object_bb[1][1] - object_bb[0][1]
	object_zz = object_bb[1][2] - object_bb[0][2]

	object_xx = object_yy = object_zz = max(object_xx, object_yy, object_zz)

	bb_extra_dist = object_xx / 2
	bb_upper[1] += bb_extra_dist
	bb_upper[2] += bb_extra_dist
	bb_lower -= bb_extra_dist

	# drawAABB([bb_lower, bb_upper], p)
	uniform_points = sample_points_in_bb_uniform(1000, bb_lower, bb_upper)

	return uniform_points


class NegPoseDataCollector(PoseDataCollector):
	def __init__(self, p_id):
		super(NegPoseDataCollector, self).__init__(p_id)
	
	def collect_neg_data_one_hook_object(self, hook_bullet_id, object_bullet_id, hook_urdf, object_urdf, hook_scaling, object_scaling, hook_world_pos):
		try:
			hook_bb = self.p.getAABB(hook_bullet_id, 0)
		except:
			hook_bb = self.p.getAABB(hook_bullet_id, -1)

		object_bb = self.p.getAABB(object_bullet_id)
		potential_pos_world = sample_neg_points_hook(hook_bb, object_bb, self.p)
		potential_quat = sample_quat_uniform(potential_pos_world.shape[0])
		ct = 0
		result_arr = []

		hook_fcl_model = fcl_load_urdf(hook_urdf)
		object_fcl_model = fcl_load_urdf(object_urdf)
		for i in range(potential_pos_world.shape[0]):
			self.p.resetBasePositionAndOrientation(object_bullet_id, potential_pos_world[i], potential_quat[i])
			self.p.changeDynamics(object_bullet_id, -1, contactStiffness=0.05, contactDamping=0.01)
			self.p.changeDynamics(hook_bullet_id, 0, contactStiffness=0.05, contactDamping=0.01)
			

			object_pos_local = potential_pos_world[i] - hook_world_pos
			object_quat_local = potential_quat[i]

			pene_dist = fcl_get_dist(hook_fcl_model, object_fcl_model, pose_transl=object_pos_local, pose_quat=object_quat_local)
			if pene_dist <= 0:
				# penetration
				continue
			if pene_dist > 0.01:
				continue
			failure = False
			for j in range(50):
				self.p.stepSimulation()
				# for tmp in self.p.getContactPoints(hook_bullet_id, object_bullet_id):
					# if tmp[8] < -0.001:
						# failure = True
						# break
				object_pos_world, object_quat = self.p.getBasePositionAndOrientation(object_bullet_id)

				if failure:
					break
				if object_pos_world[2] < 0.2 or np.linalg.norm(object_pos_world) > 5:
					failure = True
					break
				if self.check_object_touches_ground(object_bullet_id):
					failure = True
					break
				if np.linalg.norm(potential_pos_world[i] - object_pos_world) > 0.3:
					failure = True
					break
			if not failure:
				continue

			ct += 1
			# print(j)
			result_arr.append([
				hook_scaling, object_scaling, *object_pos_local, *object_quat_local 
			])
			if ct >= 50:
				break
		return result_arr

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--object_name", default='')
	parser.add_argument('--object_category')
	parser.add_argument("--sherlock", action='store_true')
	parser.add_argument("--bullet_gui", action='store_true')
	args = parser.parse_args()

	if args.sherlock:
		args.home_dir_data = '/scratch/groups/bohg/hang'
		assert args.hook_name != ''
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	neg_collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_neg')
	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_category(data_dir, args.object_category)
	p_id = bc.BulletClient(connection_mode=pybullet.GUI if args.bullet_gui else pybullet.DIRECT)
	if not os.path.exists(neg_collection_result_folder_dir):
		os.mkdir(neg_collection_result_folder_dir)
	collector = NegPoseDataCollector(p_id)
	ct = 0
	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		hook_urdf_dir = all_hook_urdf[i]
		hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)
		hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
		hook_world_pos = collector.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
		print('hook world pos offest', hook_world_pos_offset)
		
		for j, object_name in enumerate(all_object_name):
			if args.object_name != '' and args.object_name != object_name:
				continue
			object_urdf_dir = all_object_urdf[j]
			result_file_name = hook_name + '_' + object_name
			out_dir = os.path.join(neg_collection_result_folder_dir, result_file_name + '.txt')

			ct += 1
			object_bullet_id = collector.p.loadURDF(object_urdf_dir, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)
			object_scaling = collector.p.getCollisionShapeData(object_bullet_id, -1)[0][3][0] 	

			result_arr = collector.collect_neg_data_one_hook_object(hook_bullet_id, object_bullet_id, hook_urdf_dir, object_urdf_dir, hook_scaling, object_scaling, hook_world_pos)
			print(len(result_arr), result_file_name)
			with open(out_dir, 'w+') as f:
				for result in result_arr:
					f.write(comma_separated(result) + '\n')
			print(out_dir)
			collector.p.removeBody(object_bullet_id)
			if (ct + 1) % 20 == 0:
				print('reset')
				collector.p.disconnect()
				p_id = bc.BulletClient(connection_mode=pybullet.GUI if args.bullet_gui else pybullet.DIRECT)
				collector = NegPoseDataCollector(p_id)
				hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)

		collector.p.removeBody(hook_bullet_id)	

	# for j in range(20000):
		# collector.p.stepSimulation()
		# time.sleep(1./240.)