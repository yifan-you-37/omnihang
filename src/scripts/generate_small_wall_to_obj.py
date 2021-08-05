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
import xml.etree.ElementTree as ET 

import shutil

sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file
from obj_loader import *
import bullet_client as bc

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--hook_name", default='')
	args = parser.parse_args()
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(data_dir, with_small_wall=False)

	hooks_need_small_wall = load_hooks_need_small_wall(data_dir)

	wall_obj_dir = os.path.join(data_dir, 'wall', 'model_wt.obj')
	for i in range(len(all_hook_name)):
		hook_name = all_hook_name[i]
		hook_urdf = all_hook_urdf[i]
		if not hook_name == args.hook_name and args.hook_name != '':
			continue
		
		hook_cat, hook_id = split_name(hook_name)
		hook_obj_dir = os.path.join(data_dir, hook_cat, str(hook_id), 'model_meshlabserver_normalized_wt.obj')
		_, hook_scaling = get_name_scale_from_urdf(hook_urdf)

		hook_v, hook_f, hook_vn = obj_loader(hook_obj_dir)

		assert hook_vn.shape[0] == 0
		hook_bbox_max = np.max(hook_v, axis=0)
		hook_bbox_min = np.min(hook_v, axis=0)

		wall_y_length = hook_bbox_max[1] - hook_bbox_min[1]
		wall_z_length = hook_bbox_max[2] - hook_bbox_min[2]

		wall_thickness = 0.005 / float(hook_scaling)
		print(hook_scaling)
		wall_pos = np.array([0., 0., 0.])
		wall_pos[0] += wall_thickness / 2. + (hook_bbox_max[0] - hook_bbox_min[0]) / 2.
		wall_v, wall_f, wall_vn = obj_loader(wall_obj_dir)

		wall_v[:, 0] *= wall_thickness / 2.
		wall_v[:, 1] *= wall_y_length / 2.
		wall_v[:, 2] *= wall_z_length / 2.
		wall_v += wall_pos
		print('wall', wall_thickness, np.max(wall_v, axis=0) - np.min(wall_v, axis=0))
		
		all_v, all_f = merge_obj(hook_v, hook_f, wall_v, wall_f)
		all_v[:, 0] -= wall_thickness / 2.
		print(wall_thickness / 2. * hook_scaling)
		print('out bounding box', (np.max(all_v, axis=0) + np.min(all_v, axis=0))/ 2)
		out_obj_dir = os.path.join(data_dir, hook_cat, str(hook_id), 'model_small_wall.obj')
		obj_exporter(all_v, all_f, out_obj_dir)

		# print('hook bb', (np.max(hook_v, axis=0) + np.min(hook_v, axis=0)) / 2)
		# print(bbox_center[0] * hook_scaling)
		# assert np.allclose(bbox_center[0], wall_thickness / 2)

		# print(out_obj_dir)
		#output urdf file
		if hook_name in hooks_need_small_wall:
			backup_dir = os.path.join(data_dir, hook_cat, str(hook_id), 'model_concave_no_wall_ori.urdf')
			if not os.path.exists(backup_dir):
				shutil.copyfile(hook_urdf, backup_dir)
				# print(backup_dir)

		tree = ET.parse(hook_urdf)
		robot = tree.getroot()
		for mesh in robot.findall('.//mesh'):
			if 'model_' in mesh.attrib['filename']:
				mesh.set('filename', 'model_small_wall.obj')

		if not (hook_name in hooks_need_small_wall):
			out_urdf_dir = os.path.join(data_dir, hook_cat, str(hook_id), 'model_small_wall.urdf')
		else:
			out_urdf_dir = os.path.join(data_dir, hook_cat, str(hook_id), 'model_concave_no_wall.urdf')

		save_urdf(out_urdf_dir, tree)
		print(out_urdf_dir)
		# break