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
import numpy as np
import time

from sklearn.neighbors import KDTree 
from collect_pose_data import PoseDataCollector

sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file
import bullet_client as bc

from scipy.spatial import KDTree
import heapq

try:
	from mayavi import mlab as mayalab 
except:
	pass

def delete_duplicate_cp(cp_arr):
	cp_arr_after = []
	num_cp = len(cp_arr)
	for i in range(num_cp):
		flag = True
		for j in range(i+1, num_cp):
			cp_i_1 = np.array(cp_arr[i][5])
			cp_i_2 = np.array(cp_arr[i][6])
			cp_j_1 = np.array(cp_arr[j][5])
			cp_j_2 = np.array(cp_arr[j][6])
			# print(i, j, np.linalg.norm(cp_i_1 - cp_j_1), np.linalg.norm(cp_i_2 - cp_j_2))
			# print(cp_i_1, cp_i_2, cp_j_1, cp_j_2)
			if np.linalg.norm(cp_i_1 - cp_j_1) < 0.001 and np.linalg.norm(cp_i_2 - cp_j_2) < 0.001:
				flag = False
				break
		if flag:
			cp_arr_after.append(cp_arr[i])
	return cp_arr_after

def calculate_distances(neighbor_d, neighbor_i, starting_vertex):
	n_vertex = neighbor_d.shape[0]


	distances = np.ones((n_vertex)) * float('infinity') 
	distances[starting_vertex] = 0

	pq = [(0, starting_vertex)]
	while len(pq) > 0:
		current_distance, current_vertex = heapq.heappop(pq)

		# Nodes can get added to the priority queue multiple times. We only
		# process a vertex the first time we remove it from the priority queue.
		if current_distance > distances[current_vertex]:
			continue

		for neighbor, weight in zip(neighbor_i[current_vertex], neighbor_d[current_vertex]):
			distance = current_distance + weight

			# Only consider this new path if it's better than any path we've
			# already found.
			if distance < distances[neighbor]:
				distances[neighbor] = distance
				heapq.heappush(pq, (distance, neighbor))
	# print(distances[:10], np.max(distances), np.min(distances))
	return distances


def get_geo_dist(point_cloud, point_idx, tree):
	neighbor_d, neighbor_i = tree.query(point_cloud[:, :3], k=10)

	neighbor_d = neighbor_d[:, 1:]
	neighbor_i = neighbor_i[:, 1:]
	return calculate_distances(neighbor_d, neighbor_i, point_idx)
	
def nearest_neighbor_soft(point_cloud, all_point, tree, obj_scaling):
	all_val = np.zeros((point_cloud.shape[0]))
	all_maps = []

	for i in range(all_point.shape[0]):
		t_a = time.time()
		point = all_point[i]
		dist = np.linalg.norm(point_cloud[:, :3]-point.reshape(1, -1), axis=1)
		min_idx_arr = np.argsort(dist)[:5]
		angles = get_angle_batch(point_cloud[:, 3:], point_cloud[min_idx_arr[0], 3:])

		dist_normalized = dist / np.max(dist)
		dist_exp = np.exp(-2 * dist_normalized)

		angles_normalized = angles / np.max(angles)
		angles_exp = np.exp(-3 * angles_normalized)

		t_b = time.time()
		geo_dist = get_geo_dist(point_cloud, min_idx_arr[0], tree=tree)
		noninf_idx = geo_dist != np.inf
		inf_idx = geo_dist == np.inf
		geo_dist_normalized = np.zeros_like(geo_dist)
		geo_dist_normalized[noninf_idx] = geo_dist[noninf_idx] / np.max(geo_dist[noninf_idx])
		geo_dist_normalized[inf_idx] = 100
		geo_dist_exp = np.exp(-5 * geo_dist_normalized)
		t_c = time.time()

		val1 = dist_exp 
		val2 = angles_exp 
		val3 = geo_dist_exp

		# print('inf idx ct', np.sum(inf_idx))
		# print('val1 mean {} min {} max {}'.format(np.mean(val1), np.min(val1), np.max(val1)))
		# print('val2 mean {} min {} max {}'.format(np.mean(val2), np.min(val2), np.max(val2)))

		val = val2 * val3
		all_maps.append(val)
		all_val += val
		# plot_pc_s(point_cloud, val1)
		# mayalab.show()
		# plot_pc_s(point_cloud, val2)
		# mayalab.show()

		# plot_pc_s(point_cloud, val3)
		# mayalab.show()

		# plot_pc_s(point_cloud, val1 * val2)
		# mayalab.show()

		# print('val 2 * val 3')
		# plot_pc_s(point_cloud, val2 * val3 )
		# mayalab.show()

		
		# print('all')
		# plot_pc_s(point_cloud, val1 * val2 * val3)
		# mayalab.show()
		
		# import matplotlib.pyplot as plt

		# fig, axs = plt.subplots(2, 2)
		# axs[0, 0].hist(x=val1, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
		# axs[0, 0].title.set_text('l2_val')
		# axs[0, 1].hist(x=val2, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
		# axs[0, 1].title.set_text('angle_val')

		# axs[1, 0].hist(x=val3, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
		# axs[1, 0].title.set_text('geo_val')
		# axs[1, 1].hist(x=val, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
		# axs[1, 1].title.set_text('l2_val * angle_val * geo_val')

		# plt.show()
		# print(t_b - t_a, t_c - t_b)
	all_val /= np.max(all_val)

	# print('all val')
	# plot_pc_s(point_cloud, all_val)
	# mayalab.show()

	return all_val, np.array(all_maps)
	
	# print(min_idx, dist)
	# return idx

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--include_success", action='store_true')
	parser.add_argument("--bullet_gui", action='store_true')
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--object_name", default='')
	parser.add_argument("--object_category", default='')
	parser.add_argument('--more_pose', action='store_true')
	args = parser.parse_args()

	if args.object_name != '' and args.object_category == '':
		args.object_category = split_name(args.object_name)[0]
	data_dir = os.path.join(args.home_dir_data, 'geo_data')

	if not args.more_pose:
		collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
		cp_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp')
		filter_dict = None 
		
	else:
		collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_more')
		cp_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp_more')
		pose_result_more_takeoff_dict_dir = os.path.join(args.home_dir_data, 'collection_result_more_seq', 'labels', 'all_dict.json')
		filter_dict = load_json(pose_result_more_takeoff_dict_dir)
	
	mkdir_if_not(cp_result_folder_dir)
	mkdir_if_not(os.path.join(cp_result_folder_dir, 'labels'))

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_category(data_dir, args.object_category)

	if args.bullet_gui:
		p_id = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)

	collector = PoseDataCollector(p_id) 

	ct = 0

	# all_hook_name, all_hook_urdf = shuffle_two_list(all_hook_name, all_hook_urdf)
	# all_object_name, all_object_urdf = shuffle_two_list(all_object_name, all_object_urdf)

	tmp_flag = False
	for i, hook_name in enumerate(all_hook_name):
		hook_urdf_dir = all_hook_urdf[i]
		hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf_dir)

		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		hook_pc = np.load(hook_pc_dir)
		hook_tree = KDTree(hook_pc[:, :3])

		hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)
		hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
		hook_world_pos = collector.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
		
		for j, object_name in enumerate(all_object_name):
			object_urdf_dir = all_object_urdf[j]
			object_pc_dir = get_numpy_dir_from_urdf(object_urdf_dir)
			if args.object_name != '' and args.object_name != object_name:
				continue
			print(object_urdf_dir)
			print(hook_pc_dir, object_pc_dir)
			object_pc = np.load(object_pc_dir)
			object_tree = KDTree(object_pc[:, :3])
			result_file_name = hook_name + '_' + object_name
			result_file_dir = os.path.join(collection_result_folder_dir, result_file_name + '.txt')
			if not os.path.isfile(result_file_dir):
				continue

			result_np = load_result_file(result_file_dir)
			included_rows = []
			included_rows = list(range(result_np.shape[0]))

			ct += 1
			object_bullet_id = collector.p.loadURDF(object_urdf_dir, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)
			object_scaling = collector.p.getCollisionShapeData(object_bullet_id, -1)[0][3][0] 	
			cp_output_arr = []

			for k in range(result_np.shape[0]):
				if not (k in included_rows):
					continue

				tmp = result_file_name + '_' + str(k)
				if not(filter_dict is None):
					if not (tmp in filter_dict):
						continue
					if not filter_dict[tmp]:
						continue

				half_output_dir = os.path.join(cp_result_folder_dir, result_file_name + '_' + str(k))
				cp_map_object_dir = half_output_dir + '_cp_map_object.npy'
				cp_map_hook_dir = half_output_dir + '_cp_map_hook.npy'

				if os.path.exists(cp_map_object_dir) and os.path.exists(cp_map_hook_dir):
					map_object = np.load(cp_map_object_dir)
					map_hook = np.load(cp_map_hook_dir)
					if np.sum(np.isnan(map_object)) == 0 and np.sum(np.isnan(map_hook)) == 0:
						print('skip', half_output_dir)
						continue

				object_pos_local = result_np[k, -7:-4]
				object_quat_local = result_np[k, -4:]
				object_pos_world = local_pos_to_global(object_pos_local, hook_world_pos, 1.)

				collector.p.resetBasePositionAndOrientation(object_bullet_id, object_pos_world, object_quat_local)

				for _ in range(2):
					collector.p.stepSimulation()
				ct_tmp = 0
				while True:
					ct_tmp += 1
					if ct_tmp == 50:
						break
					cp_arr = collector.p.getContactPoints(object_bullet_id, hook_bullet_id)
					object_pos_global, object_quat_global = collector.p.getBasePositionAndOrientation(object_bullet_id)
					if len(cp_arr) > 0:
						break
					else:
						collector.p.stepSimulation()
				if ct_tmp == 50:
					continue

				index_object_unique = []
				index_hook_unique = []

				cp_arr = delete_duplicate_cp(cp_arr)
				# print('after', len(cp_arr))

				cp_pos_object = np.zeros((len(cp_arr), 3))
				cp_pos_hook = np.zeros((len(cp_arr), 3))

				for ii in range(len(cp_arr)):
					cp = cp_arr[ii]
					cp_pos_object[ii] = global_pos_to_local(cp[5], object_pos_global, 1., object_quat_global, collector.p)
					cp_pos_hook[ii] = global_pos_to_local(cp[6], hook_world_pos, 1.) 
				

				object_cp_map, object_cp_map_per = nearest_neighbor_soft(object_pc, cp_pos_object, tree=object_tree, obj_scaling=object_scaling)
				hook_cp_map, hook_cp_map_per = nearest_neighbor_soft(hook_pc, cp_pos_hook, tree=hook_tree, obj_scaling=hook_scaling)
				print(hook_name, object_name, k)

				
				# break
				np.save(half_output_dir + '_cp_map_object.npy', object_cp_map)
				np.save(half_output_dir + '_cp_map_hook.npy', hook_cp_map)
				np.save(half_output_dir + '_cp_map_per_object.npy', object_cp_map_per)
				np.save(half_output_dir + '_cp_map_per_hook.npy', hook_cp_map_per)

				# with open(half_output_dir + '_raw.json', 'w+') as f:
					# json.dump(cp_arr, f)
				# print(half_output_dir + '_raw.json')

			# break
			collector.p.removeBody(object_bullet_id)
			if (ct + 1) % 10 == 0:
				print('reset')
				collector.p.disconnect()

				if args.bullet_gui:
					p_id = bc.BulletClient(connection_mode=pybullet.GUI)
				else:
					p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
				collector = PoseDataCollector(p_id)
				hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)
		# break
		collector.p.removeBody(hook_bullet_id)

	# for _ in range(100000):
	# 	collector.p.stepSimulation()
	# 	print(len(collector.p.getContactPoints(hook_bullet_id, object_bullet_id)))
	# 	time.sleep(0.01)

