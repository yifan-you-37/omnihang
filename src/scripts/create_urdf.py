import os
import xml.etree.ElementTree as ET 
import sys
import numpy as np
import shutil
import argparse
import random
random.seed(2)

# unzip_dir = '/home/yifany/geo_data/hook_wall/test'

sys.path.insert(1, '../utils/')
from obj_loader import obj_loader, get_bb
from data_helper import *


# urdf_concave_template = 'templates/model_wall_hook_template.urdf'
urdf_concave_template = 'templates/model_wall_hook_template_no_wall.urdf'
# urdf_concave_template_no_wall = 'templates/model_wall_hook_template_no_wall.urdf'
urdf_convex_template_no_wall = 'templates/model_object_template_no_wall.urdf'
wall_template = 'templates/wall'

WALL_POS = np.array([0.7, 0, 1])
WALL_THICKNESS = 0.19
WALL_HOOK_POS_LOCAL = [0, 0, 0]

def set_origin(parent, pos):
	for origin in parent.findall('.//origin'):
		origin.set('xyz', ' '.join(['{:.5f}'.format(tmp) for tmp in list(pos)]))

def set_visual_collision(parent, pos, name=None, scale=None):
	for ele in parent.getchildren():
		if ele.tag in ['visual', 'collision']:
			for mesh in ele.findall('.//mesh'):
				if name is not None:
					mesh.set('filename', name)
				if scale is not None and scale != 1:
					mesh.set('scale', '{:.5f} {:.5f} {:.5f}'.format(scale, scale, scale))
			set_origin(ele, pos)

def create_urdf_concave(unzip_dir, use_labels_txt, labels_txt_dir):
	if use_labels_txt:
		obj_id_arr, obj_scaling_arr = load_labels_txt_by_keys(labels_txt_dir, ['id', 'scaling'])
	else:
		obj_id_arr = get_int_folders_in_dir(unzip_dir, int_only=True)

	for i, obj_id in enumerate(obj_id_arr):
		obj_folder = str(obj_id)
		obj_name = 'model_meshlabserver_normalized.obj'
		if use_labels_txt:
			obj_scaling = obj_scaling_arr[i]
		else:
			obj_scaling = 1.
	
		obj_dir = os.path.join(unzip_dir, obj_folder, obj_name)
		urdf_dir = os.path.join(unzip_dir, obj_folder, 'model_concave_no_wall.urdf')

		if not os.path.isfile(obj_dir):
			continue
	
		# wall_dir = os.path.join(unzip_dir, obj_folder, 'wall')
		# assert not os.path.isdir(wall_dir)
		# if os.path.isdir(wall_dir):
			# shutil.rmtree(wall_dir)
		# shutil.copytree(wall_template, wall_dir)
	
		_, bb_half_extent = get_bb(obj_dir) 
		obj_pos = np.array(WALL_HOOK_POS_LOCAL, dtype=np.float32)
		print(bb_half_extent)
		print(obj_pos[0])
		print('scaling', obj_scaling)
		obj_pos[0] = obj_pos[0] - bb_half_extent[0] * obj_scaling - WALL_THICKNESS / 2
		# print(bb_half_extent[0], obj_scaling, WALL_THICKNESS / 2)
	
	
		tree = ET.parse(urdf_concave_template)
		robot = tree.getroot()
		robot.set('name', 'obj_{}'.format(obj_id))
	
		for link in robot.findall('./link'):
			if link.attrib['name'] == 'link_hook':
				set_visual_collision(link, obj_pos, scale=obj_scaling, name=obj_name)
			elif link.attrib['name'] == 'link_wall':
				set_visual_collision(link, WALL_POS)
	
		for link in robot.findall('./joint'):
			if link.attrib['name'] == 'wall_to_hook':
				set_origin(link, WALL_POS)
	
		with open(urdf_dir, 'wb+') as f:
			f.write(ET.tostring(robot))
	
	
		print(urdf_dir)

def create_urdf_concave_no_wall_one(unzip_dir, obj_id, obj_name, urdf_dir, obj_scaling=1.):

		tree = ET.parse(urdf_concave_template_no_wall)
		robot = tree.getroot()
		robot.set('name', 'obj_{}'.format(obj_id))

		obj_dir = os.path.join(unzip_dir, str(obj_id), obj_name)
		_, bb_half_extent = get_bb(obj_dir) 

		for link in robot.findall('./link'):
			if link.attrib['name'] == 'link_hook':
				set_visual_collision(link, np.array([0, 0, bb_half_extent[2] * obj_scaling]), scale=obj_scaling, name=obj_name)
	
		with open(urdf_dir, 'wb+') as f:
			f.write(ET.tostring(robot))

def create_urdf_convex_no_wall_one(obj_id, obj_name, urdf_dir, obj_scaling=1.):

		tree = ET.parse(urdf_convex_template_no_wall)
		robot = tree.getroot()
		robot.set('name', 'obj_{}'.format(obj_id))
	
		for link in robot.findall('./link'):
			if link.attrib['name'] == 'link_object':
				set_visual_collision(link, np.array([0, 0, 0]), scale=obj_scaling, name=obj_name)
	
		with open(urdf_dir, 'wb+') as f:
			f.write(ET.tostring(robot))
		print(urdf_dir)
			
def create_urdf_concave_no_wall(unzip_dir, use_labels_txt, labels_txt_dir, random_scaling=False, scale_lb=0.1, scale_ub=1.):
	if use_labels_txt:
		obj_id_arr, obj_scaling_arr = load_labels_txt_by_keys(labels_txt_dir, ['id', 'scaling'])
	else:
		obj_id_arr = get_int_folders_in_dir(unzip_dir, int_only=True)

	for i, obj_id in enumerate(obj_id_arr):
		obj_folder = str(obj_id)
		obj_name = 'model_meshlabserver_normalized.obj'
		urdf_dir = os.path.join(unzip_dir, obj_folder, 'model_concave_no_wall.urdf')
		
		if use_labels_txt:
			obj_scaling = obj_scaling_arr[i]
		else:
			obj_scaling = 1.
		obj_dir = os.path.join(unzip_dir, obj_folder, obj_name)
		
		if os.path.isfile(obj_dir):
			if random_scaling:
				create_urdf_concave_no_wall_one(unzip_dir, obj_id, obj_name, urdf_dir, obj_scaling=random.uniform(scale_lb, scale_ub))
			else:
				create_urdf_concave_no_wall_one(unzip_dir, obj_id, obj_name, urdf_dir, obj_scaling=obj_scaling)
			print(urdf_dir)

def create_urdf_convex_no_wall(unzip_dir, random_scaling=False, scale_lb=0.1, scale_ub=1.):
	for obj_folder in os.listdir(unzip_dir):
		print(obj_folder)
		if not rep_int(obj_folder):
			continue
	
		obj_id = int(obj_folder)
		obj_name_arr = ['model_meshlabserver_normalized_v.obj', 'model_normalized_v.obj']
		urdf_dir = os.path.join(unzip_dir, obj_folder, 'model_convex_no_wall.urdf')
		obj_scaling = 1

		flag = False
		for obj_name in obj_name_arr:
			obj_dir = os.path.join(unzip_dir, obj_folder, obj_name)
			if os.path.isfile(obj_dir):
				assert not flag
				if random_scaling:
					create_urdf_convex_no_wall_one(obj_id, obj_name, urdf_dir, obj_scaling=random.uniform(scale_lb, scale_ub))
				else:
					create_urdf_convex_no_wall_one(obj_id, obj_name, urdf_dir, obj_scaling=obj_scaling)
				flag = True

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="/home/yifany/geo_data/")
	parser.add_argument('--labels_folder_dir', default='/home/yifany/geo_data/labels/')

	parser.add_argument('--category')
	parser.add_argument('--convex', action='store_true')
	parser.add_argument('--no_wall', action='store_true')
	parser.add_argument('--random_scaling', action='store_true')
	parser.add_argument('--scale_ub', default=0.1)
	parser.add_argument('--scale_lb', default=1.)

	parser.add_argument('--use_labels_txt', action='store_true')
	args = parser.parse_args()

	
	unzip_dir = os.path.join(args.data_dir, args.category)

	labels_txt_dir = os.path.join(args.labels_folder_dir, '{}_rpy_scaling.txt'.format(args.category))
	print(labels_txt_dir)
	if args.use_labels_txt:
		assert os.path.isfile(labels_txt_dir)
	
	if args.no_wall or args.convex:
		if args.convex:
			create_urdf_convex_no_wall(unzip_dir, random_scaling=args.random_scaling, scale_lb=float(args.scale_lb), scale_ub=float(args.scale_ub))
		else:
			create_urdf_concave_no_wall(unzip_dir, use_labels_txt=args.use_labels_txt, labels_txt_dir=labels_txt_dir, scale_lb=float(args.scale_lb), scale_ub=float(args.scale_ub), random_scaling=args.random_scaling)

	else:
		create_urdf_concave(unzip_dir,  use_labels_txt=args.use_labels_txt, labels_txt_dir=labels_txt_dir)

		