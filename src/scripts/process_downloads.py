import zipfile
import os
import argparse
import sys
sys.path.insert(1, '../utils/')
from obj_loader import obj_loader
from data_helper import *

def unzip_files(downloads_dir, unzip_dir):
	# unzip files 
	for file_name in os.listdir(downloads_dir):
		if file_name.endswith('.zip'):
			zip_file_dir = os.path.join(downloads_dir, file_name)
	
			# zip folder has int name
			file_number = int(os.path.splitext(file_name)[0])
	
			unzip_file_dir = os.path.join(unzip_dir, str(file_number))
	
			if os.path.isdir(unzip_file_dir):
				continue
			
			print(zip_file_dir)
			with zipfile.ZipFile(zip_file_dir, 'r') as zip_ref:
				zip_ref.extractall(unzip_file_dir)

def dae_to_obj(downloads_dir, unzip_dir):
	# convert dae to obj
	for obj_folder in get_int_folders_in_dir(unzip_dir):
		dae_dir = os.path.join(unzip_dir, obj_folder, 'model.dae')
		assert os.path.isfile(dae_dir)
		obj_dir = os.path.join(unzip_dir, obj_folder, 'model_meshlabserver.obj')
		# obj_normalized_dir = obj_dir[:-4] + '_normalized.obj'
		if not os.path.exists(obj_dir):
			print('converting', obj_dir)
			os.system('meshlabserver -i {} -o {} >/dev/null 2>&1'.format(dae_dir, obj_dir))
			# RunCmd(['meshlabserver', '-i',  dae_dir,  '-o', obj_dir, '>/dev/null 2>&1'], 10).Run()
			if os.path.exists(obj_dir):
				print('success')
		# if not os.path.exists(obj_normalized_dir):
		#     normalize_obj(obj_dir, obj_normalized_dir)
		#     print('normalized', obj_dir)

def run_vhacd(downloads_dir, unzip_dir):
	import pybullet as p
	
	# run v-hacd on obj
	for obj_folder in get_int_folders_in_dir(unzip_dir):
		obj_dir = os.path.join(unzip_dir, obj_folder, 'model_meshlabserver.obj')
		obj_v_dir = os.path.join(unzip_dir, obj_folder, 'model_vhacd.obj')
		assert os.path.isfile(obj_dir)
	
		if os.path.exists(obj_v_dir):
			continue
		p.vhacd(obj_dir, obj_v_dir,
			os.path.join(unzip_dir, obj_folder, 'vhacd_log.txt'))
		print(obj_v_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="/home/yifany/geo_data/")
	parser.add_argument('--category')
	parser.add_argument('--run_vhacd', action='store_true')
	args = parser.parse_args()

	
	downloads_dir = os.path.join(args.data_dir, args.category, 'downloads')
	unzip_dir = os.path.join(args.data_dir, args.category)

	unzip_files(downloads_dir, unzip_dir)
	dae_to_obj(downloads_dir, unzip_dir)
	if args.run_vhacd:
		run_vhacd(downloads_dir, unzip_dir)
		

