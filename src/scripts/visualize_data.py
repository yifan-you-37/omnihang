import time
import numpy as np
import random
import sys
import os
import argparse
import cv2
import zipfile
import itertools
import pybullet
import json
import numpy as np
import subprocess

sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
from bullet_helper import *
import obj_file
import bullet_client as bc

try:
	from mayavi import mlab as mayalab 
except:
	pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--cp", action='store_true')
	parser.add_argument("--single", action='store_true')
	parser.add_argument("--pose", action='store_true')
	parser.add_argument("--pose_folder_name", default='collection_result')
	parser.add_argument("--pose_idx", type=int, default=-1)

	parser.add_argument("--bullet", action='store_true')
	parser.add_argument("--pc", action='store_true')
	parser.add_argument("--more_pose", action='store_true')
	parser.add_argument("--meshlab", action='store_true')

	parser.add_argument("--hook", action='store_true')
	parser.add_argument("--object", action='store_true')
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--object_name", default='')
	parser.add_argument("--object_category", default='')
	parser.add_argument("--start_from", action='store_true')
	args = parser.parse_args()

	if args.cp:
		args.bullet = True
		args.pc = True
	if args.pose:
		args.bullet = True

	if args.object_name != '' and args.object_category == '':
		args.object_category = split_name(args.object_name)[0]
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_category(data_dir, args.object_category, with_small_wall=False)

	if args.bullet:
		if args.single:
			p = bc.BulletClient(connection_mode=pybullet.GUI)
	
	if args.hook_name != '':
		hook_idx = all_hook_name.index(args.hook_name)
		hook_urdf = all_hook_urdf[hook_idx]
		hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf)
		hook_obj_dir = get_obj_dir_from_urdf(hook_urdf)

		if args.pc:
			hook_pc = np.load(hook_pc_dir)

	if args.object_name != '':
		object_idx = all_object_name.index(args.object_name)
		object_urdf = all_object_urdf[object_idx]
		object_pc_dir = get_numpy_dir_from_urdf(object_urdf)
		object_obj_dir = get_obj_dir_from_urdf(object_urdf)

		if args.pc:
			object_pc = np.load(object_pc_dir)

	bullet_ct = 0
	flag = False

	if args.single:
		if args.hook_name != '' or args.hook:
			for i, hook_name in enumerate(all_hook_name):
				hook_urdf = all_hook_urdf[i]
				hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf)
				hook_obj_dir = get_obj_dir_from_urdf(hook_urdf)
				if args.hook_name != '' and args.hook_name != hook_name:
					if not args.start_from:
						continue
					if not flag:
						continue
				flag = True
				print(hook_urdf)
				if args.meshlab:
					subprocess.Popen(['meshlab', hook_obj_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					
				if args.bullet: 
					bullet_ct += 1
					hook_bullet_id = p.loadURDF(os.path.abspath(os.path.join(hook_urdf, '..', 'model_concave_no_wall.urdf')), basePosition=[-0.7, 0, -1], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
					p.stepSimulation()
					wall_bullet_id = make_wall(p.getAABB(hook_bullet_id, 0), np.array([0, 0, 0]), p)
					print(hook_urdf)
					# print(p.getAABB(hook_bullet_id, 0))
					input('rw')
					p.removeBody(hook_bullet_id)
					p.removeBody(wall_bullet_id)
					# hook_bullet_id = p.loadURDF(hook_urdf, basePosition=[-0.7, 0, -1], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
					# p.stepSimulation()
					# print(hook_urdf)
					# input('rw')
					# p.removeBody(hook_bullet_id)
				if args.pc:
					hook_pc = np.load(hook_pc_dir)
					plot_pc_n(hook_pc)
					plot_pc(hook_pc, color=(1, 0, 0))
					mayalab.show()
				if bullet_ct % 10 == 0 and args.bullet:
					p.disconnect()
					p = bc.BulletClient(connection_mode=pybullet.GUI)

		if args.object_name != '' or args.object_category != '' or args.object:
			for i, object_name in enumerate(all_object_name):
				object_urdf = all_object_urdf[i]
				object_pc_dir = get_numpy_dir_from_urdf(object_urdf)
				object_obj_dir = get_obj_dir_from_urdf(object_urdf)
				if args.object_name != '' and args.object_name != object_name:
					continue

				print(object_urdf)
				if args.meshlab:
					subprocess.Popen(['meshlab', object_obj_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					
				if args.bullet: 
					object_bullet_id = p.loadURDF(object_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
					input('rw')
					p.removeBody(object_bullet_id)
				if args.pc:
					object_pc = np.load(object_pc_dir)
					plot_pc_n(object_pc)
					plot_pc(object_pc, color=(1, 0, 0))
					mayalab.show()


	if args.cp:
		if args.more_pose:
			dataset_cp_dir = os.path.join(args.home_dir_data, 'dataset_cp_more')
		else:
			dataset_cp_dir = os.path.join(args.home_dir_data, 'dataset_cp')
		
	if args.cp or args.pose:
		assert args.hook_name != '' and args.object_name != ''
		if args.more_pose:
			result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_more')
		else:
			# result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
			result_folder_dir = os.path.join(args.home_dir_data, args.pose_folder_name)
	
		result_file_name = args.hook_name + '_' + args.object_name
		result_file_dir = os.path.join(result_folder_dir, result_file_name + '.txt')

		assert os.path.exists(result_file_dir)

		pose_np = load_result_file(result_file_dir)
		pose_np = pose_np[:, -7:]

		if args.bullet:
			p_env = p_Env(args.home_dir_data, gui=True, physics=True)


		if args.bullet:
			for i in range(pose_np.shape[0]):
				if args.pose_idx != -1 and args.pose_idx != i:
					continue
				pose_tmp = pose_np[i]

				if args.cp:
					result_file_name_w_id = result_file_name + '_' + str(i)
					cp_half_name = os.path.join(dataset_cp_dir, result_file_name_w_id)

					hook_cp_dir = cp_half_name + '_cp_map_hook.npy'
					object_cp_dir = cp_half_name + '_cp_map_object.npy'
					hook_cp_per_dir = cp_half_name + '_cp_map_per_hook.npy'
					object_cp_per_dir = cp_half_name + '_cp_map_per_object.npy'

					if not os.path.exists(hook_cp_dir):
						continue
				
				if args.bullet:
					p_env.load_pair_w_pose(result_file_name, pose_tmp[:3], pose_tmp[3:], aa=False)
					if args.pose:
						input('rw')
				if args.cp and args.pc:
					hook_cp_map = np.load(hook_cp_dir)
					object_cp_map = np.load(object_cp_dir)

					plot_pc_s(hook_pc, hook_cp_map)
					mayalab.show()
					plot_pc_s(object_pc, object_cp_map)
					mayalab.show()


		