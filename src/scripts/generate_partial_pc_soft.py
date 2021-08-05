import pybullet 
import time
import sys
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
# sys.path.insert(1, '../simulation/')
# from Microwave_Env import RobotEnv
from collect_pose_data import PoseDataCollector
sys.path.insert(1, '../lin_my/')
from classifier_dataset_torch import ClassifierDataset
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
from bullet_helper import *
import obj_file
import json
import bullet_client as bc
from scipy.spatial import KDTree

try:
	from mayavi import mlab as mayalab
except:
	pass
simulation_dir = '../simulation/'
sys.path.insert(0, simulation_dir)
from Microwave_Env import RobotEnv

def sample_cam_vectors(n, scaling, is_hook):
	cam_pos = sample_points_on_sphere_uniform(n) * scaling
	if is_hook:
		cam_pos[:, 0] = -1 * np.abs(cam_pos[:, 0])
	cam_up = np.zeros_like(cam_pos)

	for i in range(n):
		cam_up[i] = perpendicular_vector(cam_pos[i])
	
	return cam_pos, cam_up

def pick_best_partial_pc(p, obj_urdf, obj_pc_n, cp_map, is_hook, obj_scaling, obj_name):
	if is_hook:
		hook_offset = get_hook_wall_offset(obj_urdf)
		obj_world_pos = np.array([-0.7, 0, -1]) - np.array(hook_offset)
	else:
		obj_world_pos = [0, 0, 0]
	obj_bullet_id = p.loadURDF(obj_urdf, basePosition=obj_world_pos, baseOrientation=[0, 0, 0, 1], globalScaling=1., useFixedBase=True)

	max_partial_pc = None
	max_map_score = -1
	max_partial_pc_cam = None
	max_close_idx = None
	max_cam_pos = None
	max_cam_up = None
	# if is_hook:
	cam_pos, cam_up = sample_cam_vectors(10, 1.2 * obj_scaling, is_hook)

	# for param in check_params:
	for i in range(cam_pos.shape[0]):
		# params_dict_tmp = camera_params[param]
		params_dict = {
			'up_vector': cam_up[i] ,
			'eye_position': cam_pos[i]
		}
		if i == 10 and 1. * max_map_score / np.sum(cp_map) > 0.7:
			break

		partial_pc = p_partial_pc(p, obj_bullet_id, params_dict)
		partial_pc_tree = KDTree(partial_pc, leafsize=1000)

		# bb_len = np.max(obj_pc_n[:, :3], axis=0) - np.min(obj_pc_n[:, :3], axis=0)
		# radius = 0.2 * np.min(bb_len) if is_hook else 0.01
		radius = 0.0015 if is_hook else 0.002
		close_dist, close_idx = partial_pc_tree.query(obj_pc_n[:, :3], distance_upper_bound=radius)
		filter = np.where(close_dist != np.inf)
		close_idx = filter[0]

		# close_idx = obj_pc_tree.query_ball_point(partial_pc, r=0.01)
		# close_idx = [item for sublist in close_idx for item in sublist]
		# close_idx = np.unique(close_idx)

		partial_map_score = np.sum(cp_map[close_idx])

		# print('intersect', partial_map_score, 1. * partial_map_score / np.sum(cp_map), np.max(cp_map), np.sum(cp_map))
		

		# plot_pc(obj_pc_n, color=[1, 1, 1])
		# plot_pc(obj_pc_n[close_idx], color=[1, 0, 0])
		# plot_pc(obj_pc_n[cp_idx], color=[0.5, 0.5, 0.5])
		# print(partial_pc.shape, close_idx.shape, cp_idx.shape)
		# print(obj_scaling, radius, n_intersect / cp_idx.shape[0])
		# mayalab.savefig('test_partial_pc/{}_{}.jpg'.format(obj_name, param))
		# mayalab.clf()
		# plot_pc_s(obj_pc_n, cp_map)
		# plot_pc_n(obj_pc_n[close_idx])
		# mayalab.show()
		# plot_pc(obj_pc_n, color=[0, 0, 0])
		# plot_pc(obj_pc_n[close_idx], color=[1, 0, 0])
		# mayalab.show()

		if partial_map_score > max_map_score:
			max_partial_pc = obj_pc_n[close_idx]
			max_partial_pc_cam = partial_pc
			max_map_score = partial_map_score
			max_close_idx = close_idx
			max_cam_pos = cam_pos[i]
			max_cam_up = cam_up[i]

	p.removeBody(obj_bullet_id)
	return max_partial_pc, max_partial_pc_cam, max_close_idx, max_cam_pos, max_cam_up, 1. * max_map_score / np.sum(cp_map)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--object_name", default='')
	parser.add_argument("--start_id", type=int)
	parser.add_argument("--bullet_gui", action='store_true')
	parser.add_argument("--vis", action='store_true')
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	cp_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp')
	# cp_mat_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp_mat')

	train_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'test_list.txt')

	train_set_one_per_pair = ClassifierDataset(args.home_dir_data, train_list_dir, False, split='train', with_wall=False, one_per_pair=True)
	test_set_one_per_pair = ClassifierDataset(args.home_dir_data, test_list_dir, False, split='test', with_wall=False, one_per_pair=True)

	out_partial_pc_folder = os.path.join(args.home_dir_data, 'geo_data_partial_cp')
	out_partial_pc_pad_folder = os.path.join(args.home_dir_data, 'geo_data_partial_cp_pad')

	partial_pc_labels_folder = os.path.join(out_partial_pc_folder, 'labels')

	if not os.path.exists(out_partial_pc_folder):
		os.mkdir(out_partial_pc_folder)
	if not os.path.exists(partial_pc_labels_folder):
		os.mkdir(partial_pc_labels_folder)

	mkdir_if_not(out_partial_pc_pad_folder)

	if args.bullet_gui:
		p = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p = bc.BulletClient(connection_mode=pybullet.DIRECT)

	env = RobotEnv(0, p, [], 0, use_robot=False)
	p.removeBody(env.plane_id)
	p.removeBody(env.table_id)

	ct = 0

	overlap_dict = {}
	dict_out_dir = os.path.join(partial_pc_labels_folder, '{}_{}_overlap_dict.json'.format(
		'all' if args.hook_name == '' else args.hook_name, 'all' if args.object_name == '' else args.object_name))
	if os.path.exists(dict_out_dir):
		old_overlap_dict = load_json(dict_out_dir)
		overlap_dict.update(old_overlap_dict)
	# tmp_ct = 0
	for dataset in [train_set_one_per_pair, test_set_one_per_pair]:
		for i, batch_dict in enumerate(dataset.all_data):

			result_file_name = batch_dict['result_file_name']
			# pose_idx = int(batch_dict['pose_idx'])

			tmp_flag = False
			for pose_idx in range(8):
				cp_result_file_name = result_file_name + '_' + str(pose_idx)
				cp_map_o_dir = os.path.join(cp_result_folder_dir, cp_result_file_name +  '_cp_map_object.npy')
				cp_map_h_dir = os.path.join(cp_result_folder_dir, cp_result_file_name +  '_cp_map_hook.npy')
				if os.path.exists(cp_map_o_dir) and os.path.exists(cp_map_h_dir):
					tmp_flag = True
					break
			if not tmp_flag:
				continue
			hook_name = batch_dict['hook_name']
			if hook_name != args.hook_name and args.hook_name != '':
				continue
			object_name = batch_dict['object_name']
			if object_name != args.object_name and args.object_name != '':
				continue

			hook_urdf = train_set_one_per_pair.all_hook_dict[hook_name]['urdf']
			hook_pc_dir = train_set_one_per_pair.all_hook_dict[hook_name]['pc']

			object_urdf = train_set_one_per_pair.all_object_dict[object_name]['urdf']
			object_pc_dir = train_set_one_per_pair.all_object_dict[object_name]['pc']

			cp_map_o_dir = os.path.join(cp_result_folder_dir, cp_result_file_name +  '_cp_map_object.npy')
			cp_map_h_dir = os.path.join(cp_result_folder_dir, cp_result_file_name +  '_cp_map_hook.npy')

			half_out_dir = os.path.join(out_partial_pc_folder, result_file_name)
			half_out_pad_dir = os.path.join(out_partial_pc_pad_folder, result_file_name)
			# if os.path.exists(half_out_dir + '_object_partial_pc.npy') and os.path.exists(half_out_dir + '_hook_partial_pc.npy'):
				# print('skip', half_out_dir)
				# continue

			if (not os.path.exists(cp_map_o_dir)) or (not os.path.exists(cp_map_h_dir)):
				# tmp_ct += 1
				# print(tmp_ct, cp_map_o_dir)
				continue 
				
			cp_map_o = np.load(cp_map_o_dir)
			cp_map_h = np.load(cp_map_h_dir)

			if np.sum(np.isnan(cp_map_o)) != 0:
				print(result_file_name)
				continue
			if np.sum(np.isnan(cp_map_h)) != 0:
				print(result_file_name)
				continue
			continue
			
			hook_pc = np.load(hook_pc_dir)
			object_pc = np.load(object_pc_dir)

			hook_scaling = get_name_scale_from_urdf(hook_urdf)[1]
			object_scaling = get_name_scale_from_urdf(object_urdf)[1]

			hook_partial_pc, hook_partial_pc_cam, hook_partial_pc_idx, hook_cam_pos, hook_cam_up, hook_overlap_ratio = \
				pick_best_partial_pc(p, hook_urdf, hook_pc, cp_map_h, is_hook=True, obj_scaling=hook_scaling, obj_name=hook_name)

			object_partial_pc, object_partial_pc_cam, object_partial_pc_idx, object_cam_pos, object_cam_up, object_overlap_ratio = \
				pick_best_partial_pc(p, object_urdf, object_pc, cp_map_o, is_hook=False, obj_scaling=object_scaling, obj_name=object_name)

			if args.vis:
				print(ct, hook_partial_pc.shape, object_partial_pc.shape, 'h', hook_overlap_ratio, 'o', object_overlap_ratio)
				plot_pc_s(hook_pc, cp_map_h)
				plot_pc_n(hook_pc[hook_partial_pc_idx])
				mayalab.show()
				plot_pc(hook_pc, color=[0, 0, 0])
				plot_pc(hook_pc[hook_partial_pc_idx], color=[1, 0, 0])
				mayalab.show()

				plot_pc_s(object_pc, cp_map_o)
				plot_pc_n(object_pc[object_partial_pc_idx])
				mayalab.show()
				plot_pc(object_pc, color=[0, 0, 0])
				plot_pc(object_pc[object_partial_pc_idx], color=[1, 0, 0])
				mayalab.show()

			overlap_dict[result_file_name] = {
				'hook': hook_overlap_ratio,
				'object': object_overlap_ratio,
				'hook_cam_pos': hook_cam_pos.tolist(),
				'hook_cam_up': hook_cam_up.tolist(),
				'object_cam_pos': object_cam_pos.tolist(),
				'object_cam_up': object_cam_up.tolist(),
				'n_partial_pc_hook': hook_partial_pc.shape[0],
				'n_partial_pc_object': object_partial_pc.shape[0]
			}


			np.save(half_out_dir + '_hook_partial_pc.npy', hook_partial_pc)
			np.save(half_out_dir + '_hook_partial_pc_idx.npy', hook_partial_pc_idx)
			np.save(half_out_dir + '_object_partial_pc.npy', object_partial_pc)
			np.save(half_out_dir + '_object_partial_pc_idx.npy', object_partial_pc_idx)

			full_hook_pc, full_hook_idx = pad_pc(hook_partial_pc, hook_partial_pc_idx)
			full_object_pc, full_object_idx = pad_pc(object_partial_pc, object_partial_pc_idx)

			if args.vis:
				plot_pc(hook_pc, color=[0, 0, 0])
				plot_pc(full_hook_pc)
				mayalab.show()

				plot_pc(object_pc, color=[0, 0, 0])
				plot_pc(full_object_pc)
				mayalab.show()
				
			np.save(half_out_pad_dir + '_hook_partial_pc_pad.npy', full_hook_pc)
			np.save(half_out_pad_dir + '_hook_partial_pc_pad_idx.npy', full_hook_idx)
			np.save(half_out_pad_dir + '_object_partial_pc_pad.npy', full_object_pc)
			np.save(half_out_pad_dir + '_object_partial_pc_pad_idx.npy', full_object_idx)


			print()
			print(ct, half_out_dir, hook_partial_pc.shape, object_partial_pc.shape, 'h', hook_overlap_ratio, 'o', object_overlap_ratio)
			ct += 1
			if (ct + 1) % 10 == 0:
				p.disconnect()
				if args.bullet_gui:
					p = bc.BulletClient(connection_mode=pybullet.GUI)
				else:
					p = bc.BulletClient(connection_mode=pybullet.DIRECT)
				env = RobotEnv(0, p, [], 0, use_robot=False)
				p.removeBody(env.plane_id)
				p.removeBody(env.table_id)

			if (ct + 1) % 100 == 0:
				with open(dict_out_dir, 'w+') as f:
					json.dump(overlap_dict, f)
	with open(dict_out_dir, 'w+') as f:
		json.dump(overlap_dict, f)

	# for i, hook_name in enumerate(all_object_name):
	# 	hook_cat, hook_id = '_'.join(hook_name.split('_')[:-1]), int(hook_name.split('_')[-1])
	# 	hook_urdf = all_hook_urdf[i]
	# 	hook_offset = get_hook_wall_offset(hook_urdf)
	# 	hook_world_pos = (np.array([-0.7, 0, -1]) - np.array(hook_offset)) * 5
	# 	hook_bullet_id = p.loadURDF(hook_urdf, basePosition=hook_world_pos, baseOrientation=[0, 0, 0, 1], globalScaling=5, useFixedBase=True)
		
	# 	partial_pc = env.p_partial_pc(hook_bullet_id)
	# 	hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf)
	# 	hook_pc = np.load(hook_pc_dir) * 5

	# 	# partial_pc[:, 0] *= -1
	# 	# partial_pc[:, 2] += 1.2
	# 	print(np.mean(partial_pc, axis=0))
	# 	print(np.mean(hook_pc, axis=0))
	# 	from mayavi import mlab as mayalab
	# 	plot_pc(hook_pc)
	# 	plot_pc(partial_pc, color=(0, 1, 0))
	# 	mayalab.show()
	# 	time.sleep(1000)

		# p.removeBody(hook_bullet_id)

	# for i, object_name in enumerate(all_object_name):

	# 	object_cat, object_id = '_'.join(object_name.split('_')[:-1]), int(object_name.split('_')[-1])
	# 	if args.object_cat != '' and object_cat != args.object_cat:
	# 		continue

	# 	object_urdf = all_object_urdf[i]
	# 	object_bullet_id = p.loadURDF(object_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=True)
		
	# 	partial_pc = p_partial_pc(p, object_bullet_id, camera_params['front_left'])
	# 	print('partial pc shape', partial_pc.shape)
	# 	object_pc_dir = get_numpy_dir_from_urdf(object_urdf)
	# 	object_pc = np.load(object_pc_dir)[:, :3]
		
	# 	aabb = p.getAABB(object_bullet_id)
	# 	print('aabb',aabb[1], aabb[0])
	# 	print('object',np.max(object_pc, axis=0) - np.min(object_pc, axis=0))
	# 	drawAABB([np.min(object_pc, axis=0), np.max(object_pc, axis=0)], p)
	# 	print('partial', np.max(partial_pc, axis=0) - np.min(partial_pc, axis=0))
	# 	# drawAABB([np.min(partial_pc, axis=0), np.max(partial_pc, axis=0)], p)

	# 	# partial_pc[:, 0] *= -1
	# 	# partial_pc[:, 2] += 1.2
	# 	print(np.mean(partial_pc, axis=0))
	# 	print(np.mean(object_pc, axis=0))
	# 	from mayavi import mlab as mayalab
	# 	plot_pc(object_pc)
	# 	plot_pc(partial_pc, color=(0, 1, 0))
	# 	mayalab.show()
	# 	time.sleep(1000)

	# 	p.removeBody(hook_bullet_id)