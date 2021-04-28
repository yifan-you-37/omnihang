import pandas as pd
import numpy as np
import os
import zipfile
import glob
def rep_int(s):
	try: 
		int(s)
		return True
	except ValueError:
		return False

def mkdir_if_not(folder_dir):
	if not os.path.exists(folder_dir):
		os.mkdir(folder_dir)
def comma_separated(arr):
	return ','.join(['{:.8f}'.format(tmp) for tmp in arr])
def split_last(tmp):
	return '_'.join(tmp.split('_')[:-1]), tmp.split('_')[-1]

LABELS_CSV_DICT = {
	'id': 0,
	'url': 1,
	'rpy': [2, 3, 4],
	'scaling': 5
}
def load_labels_by_keys(labels_csv, keys):
	labels_df = pd.read_csv(labels_csv, header=None)

	ret = []
	for key in keys:
		index_arr = LABELS_CSV_DICT[key]
		arr = np.array(labels_df.iloc[:, index_arr])
		ret.append(arr)
	return ret

LABELS_TXT_DICT = {
	'id': 0,
	'rpy': [1, 2, 3],
	'scaling': 4
}

def load_labels_txt_by_keys(labels_txt_dir, keys):
	labels_df = pd.read_csv(labels_txt_dir, header=None)

	ret = []
	for key in keys:
		index_arr = LABELS_TXT_DICT[key]
		arr = np.array(labels_df.iloc[:, index_arr])
		ret.append(arr)
	return ret

def load_result_file(result_file_dir):
	if os.path.getsize(result_file_dir) == 0:
		return np.zeros((0, 16)) 
	
	result_df = pd.read_csv(result_file_dir, header=None)
	return result_df.to_numpy(dtype=np.float32)

def write_result_file(result_file_dir, result_np):
	assert len(result_np.shape) == 2
	with open(result_file_dir, 'w+') as f:
		for i in range(result_np.shape[0]):
			f.write(comma_separated(result_np[i]) + '\n')

def write_txt(out_dir, out_arr):
	with open(out_dir, 'w+') as f:
		for line in out_arr:
			f.write(line + '\n')
			
def get_int_folders_in_dir(parent_dir, int_only=False):
	int_folders = []
	for folder in os.listdir(parent_dir):
		if not rep_int(folder):
			continue
		if int_only:
			int_folders.append(folder)
		else:
			int_folders.append(os.path.join(parent_dir, folder))
	return int_folders

import subprocess as sub
import threading

class RunCmd(threading.Thread):
	def __init__(self, cmd, timeout):
		threading.Thread.__init__(self)
		self.cmd = cmd
		self.timeout = timeout

	def run(self):
		self.p = sub.Popen(self.cmd, stdout=sub.PIPE, stderr=sub.PIPE)
		self.p.wait()

	def Run(self):
		self.start()
		self.join(self.timeout)

		if self.is_alive():
			self.p.terminate()      #use self.p.kill() if process needs a kill -9
			self.join()

def read_int_txt(file_dir):
	arr = []
	with open(file_dir, 'r') as f:
		for line in f:
			flag = False
			for s in line.split(' '):
				if s.strip() == '':
					continue
				assert not flag
				num = int(s)
				arr.append(num)
				flag = True
	return arr
				
def get_urdf_dir_from_cat(cat, cat_dir, urdf_name, use_exclude_txt=True, exclude_txt_folder_dir=None, use_labels_txt=False, labels_folder_dir=None):

	exclude_id_arr = []
	if use_exclude_txt:
		assert os.path.exists(exclude_txt_folder_dir)
		exclude_txt_dir = os.path.join(exclude_txt_folder_dir, '{}.txt'.format(cat))
		assert os.path.isfile(exclude_txt_dir)
		if os.path.isfile(exclude_txt_dir):
			exclude_id_arr = read_int_txt(exclude_txt_dir)
	if use_labels_txt:
		labels_txt_dir = os.path.join(labels_folder_dir, '{}_rpy_scaling.txt'.format(cat))
		assert os.path.isfile(labels_txt_dir)
		include_id_arr = load_labels_txt_by_keys(labels_txt_dir, ['id'])[0]
	urdf_dir = []
	obj_id_arr = []

	tmp = list(os.listdir(cat_dir))
	cat_dir_list = []
	for dir in tmp:
		if rep_int(dir):
			cat_dir_list.append(dir)

	for obj_folder in sorted(cat_dir_list, key=int):
		if not rep_int(obj_folder):
			continue
		
		obj_id = int(obj_folder)

		if use_exclude_txt and obj_id in exclude_id_arr:
			continue
		
		if use_labels_txt:
			if not (obj_id in include_id_arr):
				continue
			
		urdf_file_dir = os.path.join(cat_dir, obj_folder, urdf_name)
		if os.path.exists(urdf_file_dir):
			# print(obj_folder)
			urdf_dir.append(urdf_file_dir)
			obj_id_arr.append(obj_id)
	
	return urdf_dir, obj_id_arr

# HOOK_TYPES = ['hook_wall', 'hook_wall_two_rod']
# OBJECT_TYPES = ['bag', 'cap', 'clothes_hanger', 'cooking_utensil', 'daily_object', 'headphone', 'knife', 'mug', 'racquet', 'scissor', 'wrench']
def decode_result_file_name(name):
	tmp_arr = name.split('_')
	i1, i2 = -1, -1
	
	for i, tmp in enumerate(tmp_arr):
		if rep_int(tmp):
			if i1 == -1:
				i1 = i
			elif i2 == -1:
				i2 = i
			else:
				raise "error"
	hook_name = '_'.join(tmp_arr[:i1])
	hook_id = int(tmp_arr[i1])

	object_name = '_'.join(tmp_arr[i1+1:i2])
	object_id = int(tmp_arr[i2])

	# assert hook_name in HOOK_TYPES
	# assert object_name in OBJECT_TYPES

	return hook_name, hook_id, object_name, object_id

def split_result_file_name(name):
	hook_name, hook_id, object_name, object_id = decode_result_file_name(name)
	return '{}_{}'.format(hook_name, hook_id), '{}_{}'.format(object_name, object_id)

def split_name(obj_name):
	return '_'.join(obj_name.split('_')[:-1]), int(obj_name.split('_')[-1])
	
def divide_chunks(l, n): 
	  
	# looping till length l 
	for i in range(0, len(l), n):  
		yield l[i:i + n] 
  

# HOOK_TYPES = ['hook_wall', 'hook_wall_two_rod', 'hook_standalone', 'hook_wall_horiz_rod']
HOOK_TYPES = ['hook_wall', 'hook_wall_two_rod', 'hook_wall_horiz_rod']
OBJECT_TYPES = ['bag', 'cap', 'clothes_hanger', 'cooking_utensil', 'daily_object', 'headphone', 'knife', 'mug', 'racquet', 'scissor', 'wrench']

def load_all_hooks_objects(data_dir, exclude_dir=None, labels_folder_dir=None, return_all=True, print_num=True, with_wall=False, ret_dict=False, with_small_wall=False):
	if exclude_dir is None:
		exclude_dir = os.path.join(data_dir, '..', 'exclude')
	all_hook_name = []
	all_object_name = []
	all_hook_urdf = []
	all_object_urdf = []
	for hook_category in HOOK_TYPES:
		hook_dir = os.path.join(data_dir, hook_category)
		if print_num:
			print(hook_dir)
		if '_wall' in hook_category:
			if with_wall:
				hook_urdf_arr, hook_id_arr = get_urdf_dir_from_cat(hook_category, hook_dir, 'model_concave.urdf', True, exclude_dir, False, labels_folder_dir)
			else:
				if with_small_wall:
					hook_urdf_arr, hook_id_arr = get_urdf_dir_from_cat(hook_category, hook_dir, 'model_concave_small_wall.urdf', True, exclude_dir, False, labels_folder_dir)
				else:
					hook_urdf_arr, hook_id_arr = get_urdf_dir_from_cat(hook_category, hook_dir, 'model_concave_no_wall.urdf', True, exclude_dir, False, labels_folder_dir)

		else:
			hook_urdf_arr, hook_id_arr = get_urdf_dir_from_cat(hook_category, hook_dir, 'model_concave_no_wall.urdf', True, exclude_dir, False, labels_folder_dir)
		all_hook_name += [hook_category + '_{}'.format(int(tmp)) for tmp in hook_id_arr]
		all_hook_urdf += hook_urdf_arr

	for object_category in OBJECT_TYPES:
		object_dir = os.path.join(data_dir, object_category)
		if print_num:
			print(object_dir)
		object_urdf_arr, object_id_arr = get_urdf_dir_from_cat(object_category, object_dir, 'model_convex_no_wall.urdf', True, exclude_dir)

		all_object_name += [object_category + '_{}'.format(int(tmp)) for tmp in object_id_arr]
		all_object_urdf += object_urdf_arr
		if print_num:
			print(object_category, len(object_urdf_arr))
	if print_num:
		print('hooks', len(all_hook_name))
		print('objects', len(all_object_name))
	if not return_all:
		return all_hook_name, all_object_name
	else:
		if not ret_dict:
			return all_hook_name, all_hook_urdf, all_object_name, all_object_urdf
		else:
			all_hook_dict = {
				hook_name: {
					'urdf': hook_urdf,
					'pc_ori': get_numpy_dir_from_urdf(hook_urdf),
				}	 
				for hook_name, hook_urdf in zip(all_hook_name, all_hook_urdf)}
			all_object_dict = {
				object_name: {
					'urdf': object_urdf,
					'pc_ori': get_numpy_dir_from_urdf(object_urdf)
				} 
				for object_name, object_urdf in zip(all_object_name, all_object_urdf)}
			return all_hook_dict, all_object_dict

def load_txt(file_dir):
	# with open(file_dir) as f:
	# 	content = f.readlines()
	# content = [x.strip() for x in content] 
	# return content
	return open(file_dir, 'r').read().splitlines()
def load_hooks_need_small_wall(data_dir):
	labels_dir = os.path.abspath(os.path.join(data_dir, '..', 'exclude', 'hook_need_small_wall.txt'))
	with open(labels_dir) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 
	return content

obj_category_split = {
	0: ['bag', 'cap', 'clothes_hanger'],
	1: ['cooking_utensil', 'headphone'],
	2: ['daily_object', 'knife',],
	3: ['mug'],
	4: ['racquet', 'scissor', 'wrench']
}


def load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir=None, labels_folder_dir=None, return_all=False, print_num=False, with_wall=False):
	if obj_cat_split_id < 0:
		all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(data_dir, exclude_dir, None, True, True, with_wall=with_wall)
	else:
		all_hook_name, all_hook_urdf, all_object_name_tmp, all_object_urdf_tmp = load_all_hooks_objects(data_dir, exclude_dir, None, True, True, with_wall=with_wall)
		all_object_name = []
		all_object_urdf = []
		for i, object_name in enumerate(all_object_name_tmp):
			cat_name = '_'.join(object_name.split('_')[:-1])
			if cat_name in obj_category_split[obj_cat_split_id]:
				all_object_name.append(object_name)
				all_object_urdf.append(all_object_urdf_tmp[i])
		if print_num:
			print('num object after applying category id', len(all_object_name))
	return all_hook_name, all_hook_urdf, all_object_name, all_object_urdf

def load_all_hooks_object_w_category(data_dir, object_category, with_wall=False, with_small_wall=False):
	all_hook_name, all_hook_urdf, all_object_name_tmp, all_object_urdf_tmp = load_all_hooks_objects(data_dir, with_wall=with_wall, with_small_wall=with_small_wall)
	all_object_name = []
	all_object_urdf = []
	for i, object_name in enumerate(all_object_name_tmp):
		cat_name = '_'.join(object_name.split('_')[:-1])
		if cat_name == object_category or object_category == '':
			all_object_name.append(object_name)
			all_object_urdf.append(all_object_urdf_tmp[i])
	print('num object after applying category', len(all_object_name))
	return all_hook_name, all_hook_urdf, all_object_name, all_object_urdf


def zipdir(path, ziph):
	# ziph is zipfile handle
	for root, dirs, files in os.walk(path):
		for file in files:
			# ziph.write(os.path.join(root, file),
					#    os.path.relpath(os.path.join(root, file),
									#    os.path.join(path, '..')))
			ziph.write(os.path.join(root, file), os.path.basename(path) + '_' + file)

def zipit(dir_list, zip_name):
	zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
	for dir in dir_list:
		zipdir(dir, zipf)
	zipf.close()

def complement(l, universe=None):
	"""
	Return the complement of a list of integers, as compared to
	a given "universe" set. If no universe is specified,
	consider the universe to be all integers between
	the minimum and maximum values of the given list.
	"""
	if universe is not None:
		universe = set(universe)
	else:
		universe = set(range(min(l), max(l)+1))
	return sorted(universe - set(l))

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def intersection_multiple(d):
	return list(set(d[0]).intersection(*d[1:]))
def inter_m(d):
	return list(set(d[0]).intersection(*d[1:]))
def union_m(d):
	return list(set().union(*d))
	
	
import xml.etree.ElementTree as ET 
def get_name_scale_from_urdf(urdf_dir):
	tree = ET.parse(urdf_dir)
	robot = tree.getroot()
	for mesh in robot.findall('.//mesh'):
		if 'model_' in mesh.attrib['filename']:
			if 'scale' in mesh.attrib.keys():
				return mesh.attrib['filename'], float(mesh.attrib['scale'].split(' ')[0])
			else:
				return mesh.attrib['filename'], 1.

def get_obj_dir_from_urdf(urdf_dir):
	obj_name, _ = get_name_scale_from_urdf(urdf_dir)
	return os.path.abspath(os.path.join(urdf_dir, '..', obj_name))

def get_hook_wall_offset(urdf_dir):
	tree = ET.parse(urdf_dir)
	robot = tree.getroot()
	for collision in robot.findall('.//collision'):
		flag = False
		for mesh in collision.findall('.//mesh'):
			if 'model_' in mesh.attrib['filename']:
				flag = True
				break
		if flag:
			for origin in collision.findall('.//origin'):
				ret_arr = (origin.attrib['xyz']).split(' ')
				ret_arr = [float(tmp) for tmp in ret_arr]
				return ret_arr

def get_numpy_dir_from_urdf(urdf_dir):
	obj_name, _ = get_name_scale_from_urdf(urdf_dir)
	return os.path.join(os.path.split(urdf_dir)[0], obj_name[:-4] + '_pc.npy')

def get_numpy_dir_from_name(obj_name, data_dir):
	obj_cat, obj_id = split_name(obj_name)
	if 'hook_wall' in obj_name:
		pc_ori_dir = os.path.join(data_dir, obj_cat, str(obj_id), 'model_meshlabserver_normalized_pc.npy')
	elif (not 'hook' in obj_name):
		pc_ori_dir = os.path.join(data_dir, obj_cat, str(obj_id), 'model_normalized_v_pc.npy')
		if not os.path.exists(pc_ori_dir):
			pc_ori_dir = os.path.join(data_dir, obj_cat, str(obj_id), 'model_meshlabserver_normalized_v_pc.npy')
	return pc_ori_dir

def plot_pc_n(pc_n_np, color=(1., 1., 1.)):
	from mayavi import mlab as mayalab 
	mayalab.quiver3d(pc_n_np[:, 0], pc_n_np[:, 1], pc_n_np[:, 2], pc_n_np[:, 3], pc_n_np[:, 4], pc_n_np[:, 5], color=tuple(color))
	# mayalab.show()

def plot_pc(obj_pc, color=(1., 1., 1.), scale=0.0005):
	from mayavi import mlab as mayalab 
	mayalab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=tuple(color), scale_factor=scale)

def plot_pc_both(obj_pc, pc_color=(1., 1., 1.), n_color=(1., 1., 1.)):
	from mayavi import mlab as mayalab 
	mayalab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=tuple(pc_color), scale_factor=0.0005)
	mayalab.quiver3d(pc_n_np[:, 0], pc_n_np[:, 1], pc_n_np[:, 2], pc_n_np[:, 3], pc_n_np[:, 4], pc_n_np[:, 5], color=tuple(n_color))

def plot_pc_s(obj_pc, obj_s, abs=False):
	from mayavi import mlab as mayalab 
	if abs:
		mayalab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], obj_s, scale_mode='none', scale_factor=0.0005, vmin=0, vmax=1)
	else:
		mayalab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], obj_s, scale_mode='none', scale_factor=0.0005)

	# from mayavi import mlab 
	# import matplotlib.cm as cm
	# x, y, z = obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2]
	# rgba = np.array([cm.hot(tmp) for tmp in obj_s])
	# print(rgba)
	# pts = mlab.pipeline.scalar_scatter(x, y, z) # plot the points
	# pts.add_attribute(rgba, 'colors') # assign the colors to each point
	# pts.data.point_data.set_active_scalars('colors')
	# g = mlab.pipeline.glyph(pts)
	# g.glyph.glyph.scale_factor = 0.1 # set scaling for all the points
	# g.glyph.scale_mode = 'data_scaling_off' # make all the points same size
	# mlab.show()
def idx_arr_to_mat(idx, n_dim, dtype=np.int32):
	ret_mat = np.zeros((n_dim, n_dim), dtype=dtype)

	if idx.size == 0:
		return ret_mat
	ret_mat[idx[:,0], idx[:,1]] = 1
	# ret_mat[idx[:,1], idx[:,0]] = 1
	return ret_mat

import torch
def tensor_from_npy(path, dtype=torch.float32):
	return torch.from_numpy(np.load(path)).type(dtype)

def check_object_touches_ground(object_bullet_id, p):
	object_bb = p.getAABB(object_bullet_id)
	if min(object_bb[0][2], object_bb[1][2]) < 0.01:
		return True
	return False

import random
def shuffle_two_list(a, b):
	c = list(zip(a, b))

	random.shuffle(c)
	
	a, b = zip(*c)
	return a, b
def cartesian_product_np(x, y):
  return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def get_2nd_last_dir(folder_dir, search_pattern='*'):
	files = list(filter(os.path.isfile, glob.glob(folder_dir + '/' + search_pattern)))
	files.sort(key=lambda x: os.path.getmtime(x))

	if len(files) <= 1:
		return None
	return files[-2]
def get_last_dir(folder_dir, search_pattern='*'):
	files = list(filter(os.path.isfile, glob.glob(folder_dir + '/' + search_pattern)))
	files.sort(key=lambda x: os.path.getmtime(x))

	if len(files) < 1:
		return None
	return files[-1]
	
def get_accuracy_per_category(succ_dict):
	obj_cat_arr = ['bag', 'cap', 'clothes_hanger', 'scissor', 'mug', 'cooking_utensil', 'knife', 'wrench', 'headphone', 'racquet']
	succ_dict_by_cat = {

	}
	for obj_cat in obj_cat_arr:
		succ_dict_by_cat[obj_cat] = []
	succ_dict_by_cat['others'] = []

	for result_file_name in succ_dict:
		hook_cat, _, object_cat, _ = decode_result_file_name(result_file_name)
		if object_cat in obj_cat_arr:
			succ_dict_by_cat[object_cat].append(succ_dict[result_file_name])
		else:
			succ_dict_by_cat['others'].append(succ_dict[result_file_name])

	for obj_cat in sorted(obj_cat_arr):
		print(obj_cat, np.mean(succ_dict_by_cat[obj_cat]))
	print('others', np.mean(succ_dict_by_cat['others']))


def load_json(file_dir):
	import json
	with open(file_dir) as f:
		ret_dict = json.load(f)
	return ret_dict

def save_json(file_dir, save_dict):
	import json
	with open(file_dir, 'w+') as f:
		json.dump(save_dict, f)

def save_urdf(file_dir, tree):
	with open(file_dir, 'wb+') as f:
		f.write(ET.tostring(tree.getroot()))

def plt_whitespace(plt):
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
		hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

def plt_grid(plt, img_arr, n_col, mat=False):
	import math
	n_row = math.ceil(len(img_arr) * 1. / n_col)
	fig, axs = plt.subplots(n_row, n_col)
	for i in range(n_row):
		for j in range(n_col):

			if n_row > 1:
				if mat:
					axs[i, j].imshow(img_arr[i, j])
				else:
					if (i * n_col + j) < len(img_arr):
						axs[i, j].imshow(img_arr[i * n_col + j])
				axs[i, j].axis('off')
			else:
				if mat:
					axs[j].imshow(img_arr[i, j])
				else:
					if (i * n_col + j) < len(img_arr):
						axs[j].imshow(img_arr[i * n_col + j])
				axs[j].axis('off')

def plt_save(plt, save_dir, dpi=1000):
	plt.savefig(save_dir, dpi=dpi, bbox_inches = 'tight', pad_inches =0)

def pad_pc(partial_pc, partial_pc_idx=None, n_pc=4096):
	n_partial_pc = partial_pc.shape[0]
	n_sample = n_pc - n_partial_pc

	full_pc = np.zeros((n_pc, 6))
	full_pc[:n_partial_pc] = partial_pc

	sample_idx = np.random.choice(n_partial_pc, n_sample)
	full_pc[n_partial_pc:] = partial_pc[sample_idx, :]

	if partial_pc_idx is None:
		return full_pc
	
	all_idx = np.append(partial_pc_idx, partial_pc_idx[sample_idx])
	return full_pc, all_idx

def top_k_np(a, k, sort=False):
	if sort:
		idx = np.argsort(a.ravel())[:-k-1:-1]
		top_k_idx = np.column_stack(np.unravel_index(idx, a.shape))
	else:
		idx = np.argpartition(-a.ravel(),k)[:k]
		top_k_idx = np.column_stack(np.unravel_index(idx, a.shape))
	
	if len(a.shape) == 1:
		top_k_idx = top_k_idx[:, 0]
		top_k_val = a[top_k_idx]
	else:
		top_k_val = a[top_k_idx[:, 0], top_k_idx[:, 1]]
	return top_k_val, top_k_idx
	
def print_object_cat_dict(object_cat_dict):
	from collections import Counter
	total_n_pose = 0
	total_n_pair = 0
	for object_cat in object_cat_dict:
		cat_n_pose = len(object_cat_dict[object_cat])
		cat_n_pair = len(Counter(object_cat_dict[object_cat]).keys())
		print(object_cat, 'n pose', cat_n_pose, 'n hook-object pair', cat_n_pair, cat_n_pose / cat_n_pair)
		total_n_pose += cat_n_pose
		total_n_pair += cat_n_pair
	print('total', 'n pose', total_n_pose, 'n_pair', total_n_pair, total_n_pose / total_n_pair)

# all_path_names = ['home_dir', 'home_dir_data', '']
# def get_paths(machine_name, collection_result_name='collection_result'):
# 	paths = {}
# 	if machine_name == 'sherlock':
# 		home_dir = '/home/users/yifanyou'
# 		home_dir_data = '/scratch/users/yifanyou/'
# 	elif machine_name == 'laptop':
# 		home_dir = '/home/yifany'
# 		home_dir_data = '/home/yifany'
# 	elif machine_name == 'bohg':
# 		home_dir = '/scr1/yifan'
# 		home_dir_data = '/scr1/yifan'

# 	data_dir = os.path.join(home_dir, 'geo_data')
# 	labels_folder_dir = os.path.join(data_dir, 'labels/')

# 	repo_dir = os.path.join(home_dir, 'geo-hook')
# 	exclude_dir = os.path.join(repo_dir, 'scripts/exclude')
# 	output_dir = os.path.join(repo_dir, 'scripts/{}'.format(collection_result_name))
# 	collection_result_folder_dir = os.path.join(repo_dir, 'scripts/{}'.format(collection_result_name))
# 	visualize_result_folder_dir = os.path.join(repo_dir, 'scripts/{}_visualize'.format(collection_result_name))
# 	chunk_folder_dir = os.path.join(home_dir_data, 'geo_data/misc_chunks')
# 	zip_folder_dir = os.path.join(repo_dir, 'scripts/{}_visualize_zip'.format(collection_result_name))

# 	if machine_name == 'sherlock':
# 		visualize_result_folder_dir = os.path.join('/scratch/groups/bohg/yifan', 'collection_result_visualize')
# 		zip_folder_dir = os.path.join('/scratch/groups/bohg/yifan', 'collection_result_visualize_zip')


	

if __name__ == '__main__':
	# tmp = load_labels_txt_by_keys('/home/yifany/geo_data/labels/hook_wall_horiz_rod_rpy_scaling.txt', ['id', 'rpy'])
	# print(tmp[1])
	# pass
	# unzip_dir = ''
	# get_urdf_dir_from_cat('')
	# result_df = pd.read_csv('/juno/downloads/new_hang/gif_baseline/labels.txt', header=None)
	# a = result_df.to_numpy()
	# a = np.array(a[:, 1:], dtype=np.float32)
	# print(np.mean(a, axis=0))

	pc = np.random.random((100, 3))
	# s = np.random.random((100))
	# s = np.random.random((100))
	s = np.ones((100)) * 0.3
	# s[:50] = 1
	plot_pc_s(pc, s)