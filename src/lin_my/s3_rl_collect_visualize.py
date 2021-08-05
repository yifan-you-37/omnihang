import numpy as np
import sys
import random
import os
import time
import argparse
import glob
import matplotlib.pyplot as plt
from functools import partial

try:
	from mayavi import mlab as mayalab 
except:
	pass
np.random.seed(2)

# from contact_point_dataset_torch_multi_label import MyDataset 
from cp_dataset_multi_label_multi_scale import MyDataset 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import *
from bullet_helper import *
from s2_utils import *

from simple_dataset import MyDataset
import pybullet as p

def print_helper(a):
	return '{} {} {}'.format(np.mean(a), np.max(a), np.min(a), np.std(a))

def cem_transform_pc_batch(pc_o, rotation_center_o, transl, aa):
	# TODO visualize
	if len(rotation_center_o.shape) == 1:
		rotation_center_o = rotation_center_o[np.newaxis, np.newaxis, :]
	if len(rotation_center_o.shape) == 2:
		rotation_center_o = np.expand_dims(rotation_center_o, axis=1)
	ret = transform_pc_batch(pc_o - rotation_center_o, transl, aa) + rotation_center_o
	return ret

import s3_bullet_checker_eval as bullet_checker_eval
import s3_bullet_checker as bullet_checker

def bullet_check(bi, bullet_check_one_pose, transl, aa, p_list, result_file_name, hook_urdf, object_urdf, fcl_hook_model=None, fcl_object_model=None, gui=False):
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
	tmp = input('rw')
	if tmp == 's':
		p_tmp.removeBody(hook_bullet_id_tmp)
		p_tmp.removeBody(object_bullet_id_tmp)

		return None, None, None
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
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--exp_name", default="")
	parser.add_argument("--eval_epoch", type=int, default=-1)
	parser.add_argument("--eval_ct", type=int, default=-1)
	parser.add_argument('--test_list', default='train_list')
	parser.add_argument('--train_list', default='test_list')
	parser.add_argument('--n_gt_sample', type=int, default=128)

	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--use_bullet_checker', action='store_true')

	parser.add_argument('--restrict_object_cat', default='')

	args = parser.parse_args()

	assert (args.eval_ct != -1) or (args.eval_epoch != -1)

	cp_result_folder_dir= os.path.join(args.home_dir_data,'dataset_cp')

	train_list_dir = os.path.join(cp_result_folder_dir,'labels','train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir,'labels','test_list.txt')

	runs_dir = 'runs/exp_s3'

	p_env = p_Env(args.home_dir_data, gui=True, physics=False)
	
	train_set = MyDataset(args.home_dir_data, train_list_dir, use_fcl=False, args=args)
	test_set = MyDataset(args.home_dir_data, test_list_dir, use_fcl=False, args=args)
	hook_world_pos = np.array([0.7, 0., 1])

	total_ct = 0
	for i, run_folder_dir in enumerate(glob.glob('{}/*{}'.format(runs_dir, args.exp_name))):
		# assert i == 0, run_folder_dir
		result_folder = run_folder_dir

		if args.eval_ct != -1:
			eval_file_dir_arr = glob.glob('{}/eval/*_ct_{}.json'.format(run_folder_dir, args.eval_ct))
		elif args.eval_epoch != -1:
			eval_file_dir_arr = glob.glob('{}/eval/*eval_epoch_{}_test.json'.format(run_folder_dir, args.eval_epoch))
		assert len(eval_file_dir_arr) == 1, eval_file_dir_arr
		eval_file_dir = eval_file_dir_arr[0]

		eval_result_dict = load_json(eval_file_dir)

		for j, result_file_name in enumerate(eval_result_dict):
			# if result_file_name != 'hook_wall_1_headphone_5':
			# if result_file_name != 'hook_wall_60_mug_146':
				# continue
			# if j < 15:
				# continue
			if result_file_name in train_set.all_result_file_names:
				dataset = train_set
			else:
				dataset = test_set
			
			hook_name, object_name = split_result_file_name(result_file_name)
			for eval_i, one_result_tmp in enumerate(eval_result_dict[result_file_name]):
				total_ct += 1
				# if total_ct % 10 == 0:
					# p_env.p = p_reset(p_env.p, gui=True)
				one_result = {}
				for tmp in one_result_tmp:
					if type(one_result_tmp[tmp]) == list:
						one_result[tmp] = np.array(one_result_tmp[tmp])
						if 'idx' in tmp:
							one_result[tmp] = one_result[tmp].astype(int)
					else:
						one_result[tmp] = one_result_tmp[tmp]

				pc_o = np.load(dataset.partial_pc_dir[result_file_name]['object'])[:, :3]
				pc_h = np.load(dataset.partial_pc_dir[result_file_name]['hook'])[:, :3]

				s1_transl = one_result['s1_transl']
				s1_aa = one_result['s1_aa']

				cem_init_transl = one_result['cem_init_transl']
				cem_init_aa = one_result['cem_init_aa']

				final_pred_transl = one_result['final_pred_transl']
				final_pred_aa = one_result['final_pred_aa']

				cem_out_transl = one_result['cem_out_transl']
				cem_out_aa = one_result['cem_out_aa']

				cem_rotation_center_o = one_result['cem_rotation_center_o']
				corr_idx_top_k_o = one_result['corr_idx_top_k_o']
				corr_idx_top_k_h = one_result['corr_idx_top_k_h']

				cem_elite_pose = one_result['cem_elite_pose']
				cem_elite_pose_scores = one_result['cem_elite_pose_scores']

				print('stage 1, 2', result_file_name)
				# p_env.load_pair_w_pose(result_file_name, s1_transl, s1_aa, aa=True)

				# flag = input('rw')
				# if flag == 's':
					# break
				# p_env.load_pair_w_pose(result_file_name, cem_init_transl, cem_init_aa, aa=True)
				# rotation_center_bullet_id = p_draw_ball(p_env.p, cem_rotation_center_o + hook_world_pos, radius=0.005)
				
				# bb_radius = 0.005
				# bb = [cem_init_transl - bb_radius + hook_world_pos, cem_init_transl + bb_radius + hook_world_pos]
				# drawAABB(bb, p_env.p)
				# input('cem init pose')
				# # pc_o_s1_transform = transform_pc(pc_o, s1_transl, s1_aa)
				# # plot_pc(pc_o_s1_transform)
				# # plot_pc(pc_o_s1_transform[corr_idx_top_k_o], color=[1, 0, 0], scale=0.002)

				# # plot_pc(pc_h)
				# # plot_pc(pc_h[corr_idx_top_k_h], color=[0, 1, 0], scale=0.002)
				# # plot_pc(cem_rotation_center_o[np.newaxis, :], color=[0, 0, 1], scale=0.002)
				# # mayalab.show()

				# # print('stage 3', result_file_name, one_result['succ'])
				# pc_o_cem_init = transform_pc(pc_o, cem_init_transl, cem_init_aa)
				# # # assert np.allclose( pc_h[corr_idx_top_k_h])
				# # print('object mean', np.mean(pc_o_cem_init[corr_idx_top_k_o], axis=0))
				# # print('hook mean', np.mean(pc_h[corr_idx_top_k_h], axis=0))
				# # p_env.load_pair_w_pose(result_file_name, s1_transl, s1_aa, aa=True)
				# # input('rw')
				# # p_env.load_pair_w_pose(result_file_name, cem_init_transl, cem_init_aa, aa=True)

				# # plot_pc(pc_h)
				# # plot_pc(pc_h[corr_idx_top_k_h], color=[0, 1, 0], scale=0.002)
				# # plot_pc(cem_rotation_center_o[np.newaxis, :], color=[0, 0, 1], scale=0.002)
				# # plot_pc(pc_o_cem_init)
				# # input('rw')

				# # print('cem')
				# # print(np.max(pc_h, axis=0) - np.min(pc_h, axis=0))
				# for ii in range(cem_elite_pose.shape[0]):
				# 	pose_tmp = cem_elite_pose[ii]
				# 	# pose_tmp[:3] = [0, 0, 0]
				# 	pc_o_tmp = cem_transform_pc_batch(pc_o_cem_init[np.newaxis, :, :], cem_rotation_center_o, pose_tmp[:3][np.newaxis, :], pose_tmp[3:][np.newaxis, :])
				# 	pose_transl, pose_aa = best_fit_transform(pc_o, pc_o_tmp[0])
				# 	p_env.load_pair_w_pose(result_file_name, pose_transl, pose_aa, aa=True)

				# 	# bb_center = 
				# 	print('score', cem_elite_pose_scores[ii], one_result['succ'])
				# 	print('cem pose', pose_tmp[:3], pose_tmp[3:])
				# 	input('rw')

				# 	# print(pose_tmp)
				# 	# plot_pc(pc_o_tmp[0])
				# 	# plot_pc(cem_rotation_center_o[np.newaxis, :], color=[0, 0, 1], scale=0.002)
				# 	# plot_pc(pc_h)
				# 	# plot_pc(cem_rotation_center_o[np.newaxis, :], color=[0, 0, 1])

				# 	# mayalab.show()
				if one_result['succ'] != 1:
					continue
				print('succ', one_result['succ'])

				# run bullet check
				# bullet_check_one_pose = bullet_checker_eval.check_one_pose_simple if args.use_bullet_checker else bullet_checker.check_one_pose_simple 
				# # bullet check
				# bullet_check_func = partial(
				# 	bullet_check,
				# 	bullet_check_one_pose=bullet_check_one_pose,
				# 	transl=[final_pred_transl],
				# 	aa=[final_pred_aa],
				# 	p_list=[p_env.p],
				# 	result_file_name=[result_file_name],
				# 	hook_urdf=[p_env.hook_dict[hook_name]['urdf']],
				# 	object_urdf=[p_env.object_dict[object_name]['urdf']],
				# 	fcl_hook_model=[None],
				# 	fcl_object_model=[None],
				# 	gui=True,
				# )
				# flag, _, _ = bullet_check_func(0)
				# print(final_pred_transl, final_pred_aa)
				p_env.load_pair_w_pose(result_file_name, final_pred_transl, final_pred_aa, aa=True)
				print(cem_elite_pose_scores)
				print('score', cem_elite_pose_scores[-1], one_result['succ'])
				input('rw')
				# p_env.p.removeBody(rotation_center_bullet_id)
				p_env.p.removeAllUserDebugItems()