import time
import numpy as np
import random
import sys
import os
import argparse
import cv2
import zipfile
import shutil
try:
	from mayavi import mlab as mayalab 
except:
	pass

from scipy import spatial
from collect_pose_data import PoseDataCollector

sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file

np.random.seed(42)
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(data_dir, with_small_wall=False)

	all_name = all_hook_name + all_object_name
	all_urdf = all_hook_urdf + all_object_urdf

	ManifoldPlus_dir = '/scr1/yifan/ManifoldPlus/build/manifold'
	for i in range(len(all_name)):
		name = all_name[i]
		urdf = all_urdf[i]
		obj_folder_dir = os.path.split(urdf)[0]

		# if name != 'hook_wall_two_rod_3':
			# continue
		print(urdf)
		obj_name, scaling = get_name_scale_from_urdf(urdf)
		print(urdf, scaling)


		out_pc_name = obj_name[:-4] + '_pc.npy'
		out_pc_dir = os.path.join(obj_folder_dir, out_pc_name)
		# move_pc_dir = out_pc_dir[:-4] + '_old.npy'
		# assert os.path.exists(out_pc_dir)
		# if os.path.exists(move_pc_dir):
		# 	if os.path.exists(out_pc_dir):
		# 		continue
		# else:
		# 	if os.path.exists(out_pc_dir):
		# 		shutil.move(out_pc_dir, move_pc_dir)

		obj_dir = os.path.join(obj_folder_dir, obj_name)
		if 'model_small_wall.obj' in obj_dir:
			obj_wt_dir = obj_dir
		else:
			obj_wt_dir = os.path.join(obj_folder_dir, obj_name[:-4] + '_wt.obj')
			
		if not os.path.isfile(obj_wt_dir):
			print('hello')
			os.system('{} --input {} --output {}'.format(ManifoldPlus_dir, obj_dir, obj_wt_dir))
		
		obj_wt = obj_file.OBJ(file_name=obj_wt_dir, scale=scaling)
		# pc_np_wt, normal_np_wt, _ = obj_wt.sample_points(40000, with_normal=True)

		# pc_n_np = np.load(out_pc_dir)
		# pc_np, normal_np = pc_n_np[:, :3], pc_n_np[:, 3:]
		# tree = spatial.KDTree(pc_np_wt)
		# nearest_idx = tree.query(pc_np)[1]

		# normal_np_new = normal_np_wt[nearest_idx]
		# pc_n_np = np.concatenate([pc_np, normal_np_new], axis=1)

		obj_wt = obj_file.OBJ(file_name=obj_wt_dir, scale=scaling)
		pc_np_wt, normal_np_wt, _ = obj_wt.sample_points(4096, with_normal=True)
		pc_n_np_wt = np.concatenate([pc_np_wt, normal_np_wt], axis=1)

		obj = obj_file.OBJ(file_name=obj_dir, scale=scaling)
		pc_np, normal_np, _ = obj.sample_points(4096, with_normal=True)

		# pc_n_np_tmp = np.concatenate([pc_np, normal_np], axis=1)
		# plot_pc(pc_n_np_tmp)
		# plot_pc_n(pc_n_np_tmp)
		# mayalab.show()

		# plot_pc(pc_n_np_wt)
		# plot_pc_n(pc_n_np_wt)

		print(out_pc_dir)
		np.save(out_pc_dir, pc_n_np_wt)
		# mayalab.show()
		# print(pc_np.shape, normal_np.shape)
		# print(pc_n_np.shape, out_pc_dir)
		# np.save(out_pc_dir, pc_n_np)
		# mayalab.quiver3d(pc_n_np[:, 0], pc_n_np[:, 1], pc_n_np[:, 2], pc_n_np[:, 3], pc_n_np[:, 4], pc_n_np[:, 5])
		# mayalab.show()
		# break



