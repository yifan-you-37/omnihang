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


sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import bullet_client as bc

simulation_dir = '../simulation/'
sys.path.insert(0, simulation_dir)
from Microwave_Env import RobotEnv

def sample_points_hook(hook_bb):
	bb_upper = np.max(hook_bb,axis=0)
	bb_lower = np.min(hook_bb,axis=0)
	bb_upper -= 0.05
	bb_lower -= 0.1

	bb_upper[2] += 0.2
	# p.addUserDebugLine(bb_lower,bb_upper, lineWidth=19)

	# uniform_points = sample_points_in_sphere_uniform(1000, center=(bb_upper + bb_lower) / 2, radius=np.linalg.norm(bb_upper - bb_lower)/2.)
	uniform_points = sample_points_in_bb_uniform(1000, bb_lower, bb_upper)

	filter_mask = filter_inside_bb(uniform_points, bb_lower, bb_upper)
	uniform_points = uniform_points[filter_mask]

	return uniform_points

class PoseDataCollector:
	def __init__(self, p_id):
		self.p = p_id
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		repo_dir = os.path.abspath(os.path.join(BASE_DIR, '..'))
		self.env = RobotEnv(0, self.p, [], 0, repo_dir=repo_dir, use_robot=False)
		self.p.removeBody(self.env.table_id)

	def get_hook_world_pos(self, hook_bullet_id, offset=None):
		try: 
			hook_world_pos = np.array(self.p.getLinkState(hook_bullet_id, 0)[0])
			assert offset is not None
			return hook_world_pos + offset
		except:
			hook_world_pos = np.array(self.p.getCollisionShapeData(hook_bullet_id, -1)[0][5])
			return hook_world_pos
		# print('hook_world_pos', hook_world_pos)
		
	def init_hook(self, hook_urdf_dir):
		hook_bullet_id = self.p.loadURDF(hook_urdf_dir, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=True)

		try:
			hook_scaling = self.p.getCollisionShapeData(hook_bullet_id, 0)[0][3][0]
		except:
			hook_scaling = self.p.getCollisionShapeData(hook_bullet_id, -1)[0][3][0]
		# print('scaling', hook_scaling)
		return hook_bullet_id, hook_scaling

	def check_result_file_success(self, hook_world_pos, hook_scaling, object_bullet_id, result_file_dir):
		result_np = load_result_file(result_file_dir)
		flag = False
		for i in range(result_np.shape[0]):
			object_pos_local = result_np[i, -7:-4]
			object_quat_local = result_np[i, -4:]
			object_pos_world = local_pos_to_global(object_pos_local, hook_world_pos, 1.)
			self.p.resetBasePositionAndOrientation(object_bullet_id, object_pos_world, object_quat_local)
			if not (self.check_object_touches_ground(object_bullet_id)):
				flag = True
				break
		return flag

	def run_collection(self, hook_category, object_category, hook_urdf_dir_arr, object_urdf_dir_arr, exclude_dir, labels_folder_dir, output_dir):
		for i, hook_urdf_dir in enumerate(hook_urdf_dir_arr):
			hook_id = hook_id_arr[i] 
			hook_bullet_id, hook_scaling = self.init_hook(hook_urdf_dir)
			hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
			hook_world_pos = self.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
			
			for j, object_urdf_dir in enumerate(object_urdf_dir_arr):
				object_bullet_id = self.p.loadURDF(object_urdf_dir, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)
				object_scaling = self.p.getCollisionShapeData(object_bullet_id, -1)[0][3][0] 
				object_id = object_id_arr[j]
				output_file_name = '{}_{}_{}_{}.txt'.format(hook_category, str(hook_id), object_category, str(object_id))
				output_file_dir = os.path.join(output_dir, output_file_name)
				
				# print(self.p.getCollisionShapeData(hook_bullet_id, 0)[0])
				# print(self.p.getCollisionShapeData(object_bullet_id, -1)[0])
				# print(hook_scaling, object_scaling)
				# print(self.p.getNumJoints(object_bullet_id))
				# print(self.p.getNumJoints(hook_bullet_id))
				if os.path.isfile(output_file_dir):
					# if self.check_result_file_success(hook_world_pos, hook_scaling, object_bullet_id, output_file_dir):
						# self.p.removeBody(object_bullet_id)
					# print('skipped', output_file_dir)
					continue
				# print(output_file_dir)
				self.collect_data_one_hook_object(hook_bullet_id, object_bullet_id, hook_scaling, object_scaling, hook_world_pos, output_file_dir)
	
				self.p.removeBody(object_bullet_id)
			self.p.removeBody(hook_bullet_id)

	def set_scene(self, hook_urdf_dir, object_urdf_dir, object_pos_local, object_quat_local):
		hook_bullet_id, hook_scaling = self.init_hook(hook_urdf_dir)
		hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
		hook_world_pos = self.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)

		object_pos_world = local_pos_to_global(object_pos_local, hook_world_pos, 1.)
		object_bullet_id = self.p.loadURDF(object_urdf_dir, basePosition=object_pos_world, baseOrientation=object_quat_local, globalScaling=1, useFixedBase=False)
		# print(object_urdf_dir, hook_urdf_dir)
		return hook_bullet_id, object_bullet_id

	def check_object_touches_ground(self, object_bullet_id):
		# contact_points = self.p.getContactPoints(self.env.plane_id, object_bullet_id)
		# return len(contact_points) > 0
		object_bb = self.p.getAABB(object_bullet_id)
		if min(object_bb[0][2], object_bb[1][2]) < 0.01:
			return True
		return False

	def take_photo(self, output_image_dir):
		cv2.imwrite(output_image_dir, self.env.take_pic())
		print(output_image_dir)
		# cv2.imshow('image',self.env._get_observation())
		# cv2.waitKey(0)

	def reset_simulation(self):
		# self.p.resetSimulation()
		self.env = RobotEnv(0, self.p, [], 0, use_robot=False)
		self.p.removeBody(self.env.table_id)

	def collect_data_one_hook_object(self, hook_bullet_id, object_bullet_id, hook_scaling, object_scaling, hook_world_pos, output_file_dir):
		with open(output_file_dir, 'a+') as f:
			f.write('')
		try:
			hook_bb = self.p.getAABB(hook_bullet_id, 0)
		except:
			hook_bb = self.p.getAABB(hook_bullet_id, -1)
		potential_pos_world = sample_points_hook(hook_bb)
		potential_quat = sample_quat_uniform(potential_pos_world.shape[0])

		# p.addUserDebugLine(np.array([0, 0, 0]), potential_pos[0][:3], lineWidth=19)
		success_ct = 0

		# tmp_pos = local_pos_to_global(np.array([-1.15839913,-0.17641518,-0.51256701]), hook_world_pos, 1.)
		# tmp_quat = np.array([-0.60993281,0.29887108,0.53472839,0.50271622])
		
		# self.p.resetBasePositionAndOrientation(object_bullet_id, tmp_pos, tmp_quat)
		
		# for j in range(20000):
		# 	self.p.stepSimulation()
		# 	time.sleep(1./240.)
		# return


		print(potential_pos_world.shape)
		# return
		for i in range(potential_pos_world.shape[0]):
			self.p.resetBasePositionAndOrientation(object_bullet_id, potential_pos_world[i], potential_quat[i])

			penetration = False
			for j in range(200):
				self.p.stepSimulation()
				# time.sleep(1./240.)
				object_pos_quat_world = self.p.getBasePositionAndOrientation(object_bullet_id)
				for tmp in self.p.getContactPoints(hook_bullet_id, object_bullet_id):
					if tmp[8] < -0.001:
						penetration = True
						break
				if penetration:
					break
				if object_pos_quat_world[0][2] < 0.2 or np.linalg.norm(object_pos_quat_world[0][:3]) > 5:
					break
				if self.check_object_touches_ground(object_bullet_id):
					break

			if penetration:
				continue
			if object_pos_quat_world[0][2] > 0.2 and np.linalg.norm(object_pos_quat_world[0][:3]) < 5 \
				and (not self.check_object_touches_ground(object_bullet_id)):
				object_quat_local = np.array(object_pos_quat_world[1])
				object_pos_local = global_pos_to_local(object_pos_quat_world[0][:3], hook_world_pos, 1.)
				
				object_potential_pos_local = global_pos_to_local(potential_pos_world[i], hook_world_pos, 1.)
				with open(output_file_dir, 'a+') as f:
					output_arr = [comma_separated([hook_scaling, object_scaling]), 
									comma_separated(object_potential_pos_local),
									comma_separated(potential_quat[i]),
									comma_separated(object_pos_local),
									comma_separated(object_quat_local)]
					f.write( ','.join(output_arr) + '\n')

				success_ct += 1
				print('success', success_ct)
				if success_ct >= 5:
					break
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument('--hook_category')
	parser.add_argument('--object_category')
	parser.add_argument('--bullet_gui', action='store_true')
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	output_dir = os.path.join(args.home_dir_data, 'collection_result')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	visualize_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize')
	chunk_folder_dir = os.path.join(args.home_dir_data, 'geo_data/misc_chunks')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')

	hook_dir = os.path.join(data_dir, args.hook_category)
	object_dir = os.path.join(data_dir, args.object_category)
	assert os.path.exists(hook_dir)
	assert os.path.exists(object_dir)
	print(data_dir, hook_dir)

	hook_urdf_dir_arr, hook_id_arr = get_urdf_dir_from_cat(args.hook_category, hook_dir, 'model_concave.urdf', True, exclude_dir, True, labels_folder_dir)
	object_urdf_dir_arr, object_id_arr = get_urdf_dir_from_cat(args.object_category, object_dir, 'model_convex_no_wall.urdf', True, exclude_dir)

	print('num hook loaded:', len(hook_id_arr))
	print('num object loaded:', len(object_id_arr))

	if args.bullet_gui:
		p_id = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)

	collector = PoseDataCollector(p_id)
	collector.run_collection(args.hook_category, args.object_category, hook_urdf_dir_arr, object_urdf_dir_arr, exclude_dir, labels_folder_dir, output_dir)