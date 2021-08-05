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

import xml.etree.ElementTree as ET 

sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import bullet_client as bc

simulation_dir = '../simulation/'
sys.path.insert(0, simulation_dir)
from Env import RobotEnv

def sample_points_hook(hook_bb, object_bb, p, more_pose=False):
	bb_upper = np.max(hook_bb,axis=0)
	bb_lower = np.min(hook_bb,axis=0)

	object_xx = object_bb[1][0] - object_bb[0][0]
	object_yy = object_bb[1][1] - object_bb[0][1]
	object_zz = object_bb[1][2] - object_bb[0][2]

	object_xx = object_yy = object_zz = max(object_xx, object_yy, object_zz)
	# bb_upper[0] -= min(object_xx / 2., 0.05 * 0.3)
	bb_upper[1] += max(object_yy / 2., 0.03)
	bb_upper[2] += max(object_zz / 2., 0.06)

	bb_lower[0] -= max(object_xx / 2., 0.03)
	bb_lower[1] -= max(object_yy / 2., 0.03)
	bb_lower[2] -= max(object_zz / 2., 0.03)

	# drawAABB([bb_lower, bb_upper], p)
	if more_pose:
		uniform_points = sample_points_in_bb_uniform(1000 * 10, bb_lower, bb_upper)
	else:
		uniform_points = sample_points_in_bb_uniform(1000, bb_lower, bb_upper)
	
	filter_mask = filter_inside_bb(uniform_points, bb_lower, bb_upper)
	uniform_points = uniform_points[filter_mask]

	return uniform_points

class PoseDataCollector:
	def __init__(self, p_id):
		self.p = p_id
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		self.repo_dir = os.path.abspath(os.path.join(BASE_DIR, '..'))
		self.env = RobotEnv(0, self.p, [], 0, repo_dir=self.repo_dir, use_robot=False)

	def get_hook_world_pos(self, hook_bullet_id, offset=None):
		try: 
			hook_world_pos = np.array(self.p.getLinkState(hook_bullet_id, 0)[0])
			assert offset is not None
			return hook_world_pos + offset
		except:
			hook_world_pos = np.array(self.p.getCollisionShapeData(hook_bullet_id, -1)[0][5])
			return hook_world_pos
		
	def init_hook(self, hook_urdf_dir):
		hook_bullet_id = self.p.loadURDF(hook_urdf_dir, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=True)

		try:
			hook_scaling = self.p.getCollisionShapeData(hook_bullet_id, 0)[0][3][0]
		except:
			hook_scaling = self.p.getCollisionShapeData(hook_bullet_id, -1)[0][3][0]
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

	def make_tmp_urdf(self, object_urdf, object_scaling):
		tree = ET.parse(object_urdf)
		robot = tree.getroot()
		for mesh in robot.findall('.//mesh'):
			if 'model_' in mesh.attrib['filename']:
				mesh.set('scale', '{:.7f} {:.7f} {:.7f}'.format(object_scaling[0], object_scaling[1], object_scaling[2]))

		out_urdf = object_urdf[:-5] + '_tmp.urdf'
		with open(out_urdf, 'wb+') as f:
			f.write(ET.tostring(robot))
			
		return out_urdf

	def run_collection(self, run_hook_name, run_object_category, all_hook_name, all_hook_urdf, all_object_name, all_object_urdf, output_dir, args):
		total_ct = 0
		result_labels_dict = {}

		if args.object_name == '':
			out_labels_dir = os.path.join(output_dir, 'labels', '{}_{}.json'.format(run_hook_name, run_object_category))
		else:
			out_labels_dir = os.path.join(output_dir, 'labels', '{}_{}_{}.json'.format(run_hook_name, run_object_category, args.object_name))

		if (not args.no_vary_scale) and args.more_pose:
			scale_dict_dir = os.path.join(output_dir, '..', 'collection_result_vary_scale', 'labels', 'scale.json')
			assert os.path.exists(scale_dict_dir), scale_dict_dir
			scale_dict = load_json(scale_dict_dir)		

		for i, hook_urdf_dir in enumerate(all_hook_urdf):
			hook_name = all_hook_name[i]
			if hook_name != run_hook_name:
				continue
			
			hook_bullet_id, hook_scaling = self.init_hook(hook_urdf_dir)
			hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
			hook_world_pos = self.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
			
			hook_bb = self.p.getAABB(hook_bullet_id, 0)
			wall_bullet_id = make_wall(hook_bb, hook_world_pos, self.p, wall_size=0.5)
			
			for j, object_urdf_dir in enumerate(all_object_urdf):
				object_name = all_object_name[j]
				if args.object_name != '' and object_name != args.object_name:
					continue
				result_file_name = '{}_{}'.format(hook_name, object_name)
				
				output_file_pos_dir = os.path.join(output_dir, result_file_name + '.txt')
				if os.path.isfile(output_file_pos_dir):
					continue
					
				object_scaling = get_name_scale_from_urdf(object_urdf_dir)[1]
				scaling_low = object_scaling * 0.6
				scaling_high = object_scaling * 1.4


				if args.no_vary_scale:
					n_sample = 1
					scaling_sample = np.ones((1, 3)) * object_scaling
				else:
					if args.more_pose:
						if result_file_name in scale_dict:
							scale_dict_tmp = scale_dict[result_file_name]
							n_sample = len(scale_dict_tmp)
							scaling_sample = []
							for ii in range(n_sample):
								scaling_sample.append(scale_dict_tmp[str(ii+1)]['object_scale_abs'])
							scaling_sample = np.stack(scaling_sample, axis=0)
							print(scaling_sample, result_file_name)
						else:
							continue
					else:
						n_sample = 5
						scaling_sample = np.random.uniform(scaling_low, scaling_high, (5, 3))


				output_arr = []
				for ii in range(n_sample):
					object_scaling_tmp = scaling_sample[ii]
					object_scaling_tmp_relative = object_scaling_tmp / object_scaling
					object_urdf_tmp_dir = self.make_tmp_urdf(object_urdf_dir, object_scaling_tmp)
					print(result_file_name)

					object_bullet_id = self.p.loadURDF(object_urdf_tmp_dir, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)

					output_arr += self.collect_data_one_hook_object(hook_bullet_id, wall_bullet_id, object_bullet_id, hook_scaling, object_scaling_tmp, object_scaling_tmp_relative, hook_world_pos, output_file_pos_dir, None, more_pose=args.more_pose)
			
					self.p.removeBody(object_bullet_id)
				result_labels_dict[result_file_name] = len(output_arr)
				if len(output_arr) > 0:
					with open(output_file_pos_dir, 'a+') as f:
						for one_line in output_arr:
							f.write(','.join(one_line) + '\n')
				save_json(out_labels_dir, result_labels_dict)
				total_ct += 1
				
				if total_ct % int(25 / n_sample) == 0:
					print('reset')
					self.p.disconnect()
					if args.bullet_gui:
						self.p = bc.BulletClient(connection_mode=pybullet.GUI)
					else:
						self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
					self.env = RobotEnv(0, self.p, [], 0, repo_dir=self.repo_dir, use_robot=False)

					hook_bullet_id, hook_scaling = self.init_hook(hook_urdf_dir)
					wall_bullet_id = make_wall(hook_bb, hook_world_pos, self.p, wall_size=0.5)

			self.p.removeBody(hook_bullet_id)
			self.p.removeBody(wall_bullet_id)

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

	def reset_simulation(self):
		self.env = RobotEnv(0, self.p, [], 0, use_robot=False)
		self.p.removeBody(self.env.table_id)

	def collect_data_one_hook_object(self, hook_bullet_id, wall_bullet_id, object_bullet_id, hook_scaling, object_scaling, object_scaling_relative, hook_world_pos, output_file_pos_dir, output_file_neg_dir, more_pose=False):
		try:
			hook_bb = self.p.getAABB(hook_bullet_id, 0)
		except:
			hook_bb = self.p.getAABB(hook_bullet_id, -1)
		potential_pos_world = sample_points_hook(hook_bb, self.p.getAABB(object_bullet_id, -1), self.p, more_pose=more_pose)
		potential_quat = sample_quat_uniform(potential_pos_world.shape[0])

		success_ct = 0

		output_arr = []
		for i in range(potential_pos_world.shape[0]):
			self.p.resetBasePositionAndOrientation(object_bullet_id, potential_pos_world[i], potential_quat[i])
			penetration = False
			for j in range(200):
				self.p.stepSimulation()
				# time.sleep(1./240.)
				object_pos_quat_world = self.p.getBasePositionAndOrientation(object_bullet_id)
				for tmp in self.p.getContactPoints(wall_bullet_id, object_bullet_id):
					if tmp[8] < -0.001:
						penetration = True
						break
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
			# if True:
				object_quat_local = np.array(object_pos_quat_world[1])
				object_pos_local = global_pos_to_local(object_pos_quat_world[0][:3], hook_world_pos, 1.)
				
				object_potential_pos_local = global_pos_to_local(potential_pos_world[i], hook_world_pos, 1.)
				output_arr.append([comma_separated([hook_scaling, object_scaling[0], object_scaling[1], object_scaling[2]]), 
								comma_separated(object_scaling_relative),
								comma_separated(object_potential_pos_local),
								comma_separated(potential_quat[i]),
								comma_separated(object_pos_local),
								comma_separated(object_quat_local)])

				success_ct += 1
				print('success', success_ct)
				if more_pose:
					if success_ct >= 5 * 10:
						break
				else:
					if success_ct >= 5:
						break
		return output_arr
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument('--hook_name')
	parser.add_argument('--object_category')
	parser.add_argument('--object_name', default='')
	parser.add_argument('--bullet_gui', action='store_true')
	parser.add_argument('--no_vary_scale', action='store_true')
	parser.add_argument('--more_pose', action='store_true')
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	if args.no_vary_scale:
		if args.more_pose:
			output_dir = os.path.join(args.home_dir_data, 'collection_result_more')
		else:
			assert False
	else:
		if args.more_pose:
			output_dir = os.path.join(args.home_dir_data, 'collection_result_vary_scale_more')
		else:
			output_dir = os.path.join(args.home_dir_data, 'collection_result_vary_scale')

	if not args.no_vary_scale:
		assert args.more_pose

	mkdir_if_not(output_dir)
	mkdir_if_not(os.path.join(output_dir, 'labels'))

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_category(data_dir, args.object_category)

	if args.bullet_gui:
		p_id = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)

	collector = PoseDataCollector(p_id)
	collector.run_collection(args.hook_name, args.object_category, all_hook_name, all_hook_urdf, all_object_name, all_object_urdf, output_dir, args=args)