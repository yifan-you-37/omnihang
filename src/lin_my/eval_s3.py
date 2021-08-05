import numpy as np
import sys
import random
from simple_dataset import MyDataset 
import os
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import *
from bullet_helper import *
from collision_helper import *

from mp_utils import *
from functools import partial


def bullet_check(bi, bullet_check_one_pose, transl, aa, p_list, result_file_name, hook_urdf, object_urdf, fcl_hook_model=None, fcl_object_model=None, gui=False):
	transl = np.array(transl)
	aa = np.array(aa)
	quat_tmp = quaternion_from_angle_axis(aa[bi])
	transl_tmp = transl[bi]
	hook_world_pos = np.array([0.7, 0., 1])
	p_tmp = p_list[bi]
	if not p_tmp.isConnected():
		p_list[bi] = p_reset_multithread(bi, p_list, gui=gui)
	p_enable_physics(p_tmp)

	hook_bullet_id_tmp = p_tmp.loadURDF(hook_urdf[bi], basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
	object_bullet_id_tmp = p_tmp.loadURDF(object_urdf[bi], basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=False)

	p_tmp.changeDynamics(hook_bullet_id_tmp, -1, contactStiffness=1.0, contactDamping=0.01)
	p_tmp.changeDynamics(hook_bullet_id_tmp, 0, contactStiffness=0.5, contactDamping=0.01)
	p_tmp.changeDynamics(object_bullet_id_tmp, -1, contactStiffness=0.05, contactDamping=0.01)

	p_tmp.resetBasePositionAndOrientation(object_bullet_id_tmp, transl_tmp + hook_world_pos, quat_tmp)
	p_tmp.resetBaseVelocity(object_bullet_id_tmp, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

	flag, final_pose = bullet_check_one_pose(
		p_tmp, 
		hook_world_pos, 
		hook_bullet_id_tmp, 
		object_bullet_id_tmp, 
		transl_tmp, 
		quat_tmp,
		hook_urdf[bi],
		object_urdf[bi],
		fcl_hook_model[bi],
		fcl_object_model[bi])
	print('{} done {}'.format(bi, flag))
	p_tmp.removeBody(hook_bullet_id_tmp)
	p_tmp.removeBody(object_bullet_id_tmp)

	return flag, final_pose[:3], final_pose[3:]

import s3_bullet_checker_eval as bullet_checker_eval

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--result_file_dir")
	parser.add_argument("--result_file_name")
	parser.add_argument("--output_folder")
	parser.add_argument("--bullet_gui", action='store_true')
	parser.add_argument("--mode", default='s123')
	parser.add_argument('--bohg4', action='store_true')

	args = parser.parse_args()

	mode = args.mode 
	if args.bohg4:
		args.home_dir_data = '/scr1/yifan/hang'

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	all_hook_dict, all_object_dict = load_all_hooks_objects(data_dir, ret_dict=True)
	
	result_dict_all = load_json(args.result_file_dir)

	result_file_name = args.result_file_name


	output_file_dir = os.path.join(args.output_folder, '{}.json'.format(result_file_name))
	hook_name, object_name = split_result_file_name(result_file_name)
	hook_urdf = all_hook_dict[hook_name]['urdf']
	object_urdf = all_object_dict[object_name]['urdf']

	result_dict = result_dict_all[result_file_name]

	hook_world_pos = np.array([0.7, 0, 1])
	p = p_reset(None, gui=args.bullet_gui)

	bullet_check_one_pose = bullet_checker_eval.check_one_pose_simple  

	fcl_hook_model = None
	fcl_object_model = None
	
	bullet_check_func = partial(
		bullet_check,
		bi=0,
		bullet_check_one_pose=bullet_check_one_pose,
		# transl=final_pred_transl,
		# aa=final_pred_aa,
		p_list=[p],
		result_file_name=[result_file_name],
		hook_urdf=[hook_urdf],
		object_urdf=[object_urdf],
		fcl_hook_model=[fcl_hook_model],
		fcl_object_model=[fcl_object_model],
		gui=args.bullet_gui,
	)

	out_dict = []

	s1_transl = result_dict[0]['s1_transl']
	s1_aa = result_dict[0]['s1_aa']
	s2_transl = result_dict[0]['cem_init_transl']
	s2_aa = result_dict[0]['cem_init_aa']

	best_cem_score = 0
	s3_transl = None
	s3_aa = None
	for i, one_result in enumerate(result_dict):	
		cem_score = float(one_result['cem_elite_pose_scores'][-1])
		if cem_score > best_cem_score:
			best_cem_score = cem_score
			s3_transl = one_result['final_pred_transl']
			s3_aa = one_result['final_pred_aa']
	
	if mode == 's123':
		s1_flag, _, _ = bullet_check_func(transl=[s1_transl], aa=[s1_aa])
		s2_flag, _, _ = bullet_check_func(transl=[s2_transl], aa=[s2_aa])
	
	s3_flag, _, _ = bullet_check_func(transl=[s3_transl], aa=[s3_aa])

	if mode == 's123':
		out_dict = {
			's1_flag': s1_flag,
			's2_flag': s2_flag,
			's3_flag': s3_flag,
		}
	elif mode == 's13':
		out_dict = {
			's3_flag': s3_flag,
		}

	save_json(output_file_dir, out_dict)


