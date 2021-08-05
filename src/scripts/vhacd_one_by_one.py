import pybullet as p
import time
import sys
import argparse
# import matplotlib.pyplot as plt
import os
# sys.path.insert(1, '../simulation/')
# from Microwave_Env import RobotEnv

sys.path.insert(1, '../utils/')
from data_helper import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="/home/yifany/geo_data/")
	parser.add_argument('--category')
	parser.add_argument('--model_name')
	parser.add_argument('--start', type=int)
	args = parser.parse_args()

	unzip_dir = os.path.join(args.data_dir, args.category)

	for obj_id in range(args.start, args.start+200):
		obj_folder = str(obj_id)

		obj_dir = os.path.join(unzip_dir, obj_folder, '{}.obj'.format(args.model_name))
		obj_v_dir = os.path.join(unzip_dir, obj_folder, '{}_v.obj'.format(args.model_name))
		if os.path.isfile(obj_dir):
			if os.path.exists(obj_v_dir):
				continue

				
			print(obj_v_dir)
			RunCmd(['python', 'to_vhacd.py',  obj_dir], 60).Run()
			if os.path.exists(obj_v_dir):
				print('success')			
			# p.vhacd(obj_dir, obj_v_dir,
				# os.path.join(unzip_dir, obj_folder, 'vhacd_log.txt'))
				