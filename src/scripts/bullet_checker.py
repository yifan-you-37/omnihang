import sys
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import bullet_client as bc
from coord_helper import * 
from data_helper import * 
import time

def init_p(p):
    p.setPhysicsEngineParameter(enableConeFriction=1)
    p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
    p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
    p.setPhysicsEngineParameter(numSolverIterations=40)
    p.setPhysicsEngineParameter(numSubSteps=40)
    p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
    p.setPhysicsEngineParameter(enableFileCaching=0)

    p.setTimeStep(1 / 100.0)
    p.setGravity(0,0,-9.81)

class BulletChecker:
	def __init__(self, gui=False):
		if gui:
			self.p = bc.BulletClient(connection_mode=pybullet.GUI)
		else:
			self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)	
		init_p(self.p)
		self.gui = gui
		
	def check_one_pose(self, hook_urdf, object_urdf, object_pose, ct):

		if ct % 10 == 0:
			print('reset bullet env')
			self.p.disconnect()
			if self.gui:
				self.p = bc.BulletClient(connection_mode=pybullet.GUI)
			else:
				self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)	
			init_p(self.p)

		hook_offset = get_hook_wall_offset(hook_urdf)
		hook_world_pos = np.array([0.7, 0, 1]) + hook_offset

		ori_object_pos, ori_object_quat = object_pose[:3], object_pose[3:]
		hook_bullet_id = self.p.loadURDF(hook_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
		object_bullet_id = self.p.loadURDF(object_urdf, basePosition=ori_object_pos + hook_world_pos, baseOrientation=ori_object_quat)
		
		failure = False
		for i in range(150):
			self.p.stepSimulation()
			
			object_pos_world, object_quat = self.p.getBasePositionAndOrientation(object_bullet_id)
			object_pos = object_pos_world - hook_world_pos

			if i%5 == 0:
				#check overlap of model with hook
				for tmp in self.p.getClosestPoints(hook_bullet_id, object_bullet_id, 0.1):
					if tmp[8] < -0.001:
						failure = True
						break
				if failure:
					break
				
				# if object center too low or object too far away
				if object_pos_world[2] < 0.2 or np.linalg.norm(object_pos_world) > 5:
					failure = True
					break
				
				# if touches ground
				if check_object_touches_ground(object_bullet_id, self.p):
					failure = True
					break
	
				# too much change in pos
				if np.linalg.norm(ori_object_pos - object_pos) > 0.1:
					failure = True
					break
			# too much change in quat

		self.p.removeBody(hook_bullet_id)
		self.p.removeBody(object_bullet_id)

		return (not failure)

	
# def check_one_pose(hook_urdf, object_urdf, object_pose, p):
	# hook_offset = get_hook_wall_offset(hook_urdf)
	# hook_world_pos = np.array([0.7, 0, 1]) + hook_offset

	# ori_object_pos, ori_object_quat = object_pose[:3], object_pose[3:]
	# hook_bullet_id = p.loadURDF(hook_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
	# object_bullet_id = p.loadURDF(object_urdf, basePosition=ori_object_pos + hook_world_pos, baseOrientation=ori_object_quat)
	
	# failure = False
	# for i in range(200):
	# 	p.stepSimulation()
		
	# 	object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
	# 	object_pos = object_pos_world - hook_world_pos

	# 	#check overlap of model with hook
	# 	for tmp in p.getContactPoints(hook_bullet_id, object_bullet_id):
	# 		if tmp[8] < -0.001:
	# 			failure = True
	# 			break
	# 	if failure:
	# 		break
		
	# 	# if object center too low or object too far away
	# 	if object_pos_world[2] < 0.2 or np.linalg.norm(object_pos_world) > 5:
	# 		failure = True
	# 		break
		
	# 	# if touches ground
	# 	if check_object_touches_ground(object_bullet_id, p):
	# 		failure = True
	# 		break

	# 	# too much change in pos
	# 	if np.linalg.norm(ori_object_pos - object_pos) > 0.3:
	# 		failure = True
	# 		break

	# 	# too much change in quat

	# p.removeBody(hook_bullet_id)
	# p.removeBody(object_bullet_id)

	# return (not failure)



if __name__ == '__main__':
	import argparse
	import pybullet
	import random
	import time

	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--use_labeled_data", action='store_true')
	parser.add_argument("--sherlock", action='store_true')
	parser.add_argument("--hook_name", default='')
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
	output_dir = os.path.join(args.home_dir_data, 'collection_result')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	visualize_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize')
	chunk_folder_dir = os.path.join(args.home_dir_data, 'geo_data/misc_chunks')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')
	zip_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize_zip')
	seq_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_seq')

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, labels_folder_dir, True, True)

	# p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
	# init_p(p_id)
	bullet_checker = BulletChecker(True)

	ct = 0

	all_hook_name, all_hook_urdf = shuffle_two_list(all_hook_name, all_hook_urdf)
	all_object_name, all_object_urdf = shuffle_two_list(all_object_name, all_object_urdf)

	start_time = None
	for i, hook_name in enumerate(all_hook_name):
		hook_urdf = all_hook_urdf[i]

		for j, object_name in enumerate(all_object_name):
			object_urdf = all_object_urdf[j]

			result_file_name = hook_name + '_' + object_name
			result_file_dir = os.path.join(collection_result_folder_dir, result_file_name + '.txt')
			if not os.path.isfile(result_file_dir):
				continue
			n_sample = 10
			sample_quat = sample_quat_uniform(n_sample)
			sample_pos = sample_points_in_bb_uniform(n_sample, np.array([-0.2, -0.2, -0.2]), np.array([-0.05, 0.2, 0.2]))[:n_sample]
			sample_pose = np.concatenate((sample_pos, sample_quat), axis=1)

			for k in range(n_sample):
				if start_time is None:
					start_time = time.time()
				if (bullet_checker.check_one_pose(hook_urdf, object_urdf, sample_pose[k], ct)):
					print(hook_urdf, object_urdf, sample_pose[k])
				ct += 1

			# result_np = load_result_file(result_file_dir)
			# for k in range(result_np.shape[0]):
			# 	if start_time is None:
			# 		start_time = time.time()
			# 	print(bullet_checker.check_one_pose(hook_urdf, object_urdf, result_np[k, -7:], ct))
			# 	ct += 1
			
				# if (ct + 1) % 10 == 0:
					# print('reset bullet env')
					# p_id.disconnect()
					# p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
					# init_p(p_id)
					# print((time.time() - start_time) / ct)
