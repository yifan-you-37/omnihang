import numpy as np
import sys
import random
import os
import time
import argparse
import glob
import matplotlib.pyplot as plt

try:
	from mayavi import mlab as mayalab 
except:
	pass
np.random.seed(2)

# from contact_point_dataset_torch_multi_label import MyDataset 
from hang_dataset import MyDataset 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import *
from bullet_helper import *
from s2_utils import *

import pybullet as p

K = 128


def plot_corr(pc_o, pc_h, pose_transl, pose_quat, cp_top_k_idx_o, cp_top_k_idx_h, corr, aa=False):
	corr = np.reshape(corr, (K, K))
	pc_o_transformed = transform_pc(pc_o, pose_transl, pose_quat, aa=aa)
	
	# plot_pc(pc_h)
	# plot_pc(pc_o_transformed)

	top_k_corr, top_k_corr_idx = top_k_np(corr, 512, sort=True)
	top_k_corr_idx_o = top_k_corr_idx[:, 0]
	top_k_corr_idx_h = top_k_corr_idx[:, 1]

	# print('top k corr mean', np.mean(top_k_corr), np.max(top_k_corr), np.min(top_k_corr))
	# plot_pc_s(pc_o_transformed[cp_top_k_idx_o][top_k_corr_idx_o], top_k_corr)
	# plot_pc_s(pc_h[cp_top_k_idx_h][top_k_corr_idx_h], top_k_corr)
	# mayalab.show()

	plot_pc(pc_h)
	plot_pc(pc_o_transformed)
	
	partial_pc_o = pc_o_transformed[cp_top_k_idx_o][top_k_corr_idx_o[:3]]
	partial_pc_h = pc_h[cp_top_k_idx_h][top_k_corr_idx_h[:3]]
	plot_pc(partial_pc_o, color=(0, 1, 0), scale=0.002)
	plot_pc(partial_pc_h, color=(0, 0, 1), scale=0.002)
	mayalab.show()

	rotation_center = np.mean(partial_pc_h - partial_pc_o, axis=0)
	# plot_pc(pc_h)
	# plot_pc(pc_o_transformed + rotation_center[np.newaxis, :])
	# plot_pc(partial_pc_o + rotation_center[np.newaxis, :], color=(0, 1, 0), scale=0.002)
	# plot_pc(partial_pc_h, color=(0, 0, 1), scale=0.002)

	# mayalab.show()
	
	# plot_pc(pc_o_transformed[cp_top_k_idx_o][top_k_np(corr[:, 0], 5)[1]], scale=0.002, color=(1, 0, 0))
	return rotation_center[:3]
def print_helper(a):
	return '{} {} {}'.format(np.mean(a), np.max(a), np.min(a), np.std(a))
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--exp_name", default="")
	parser.add_argument("--eval_epoch", type=int, default=-1)
	parser.add_argument("--eval_ct", type=int, default=-1)
	parser.add_argument('--test_list', default='test_list')
	parser.add_argument('--n_gt_sample', type=int, default=128)

	parser.add_argument('--restrict_object_cat', default='')

	args = parser.parse_args()

	assert (args.eval_ct != -1) or (args.eval_epoch != -1)
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	hook_dict, object_dict = load_all_hooks_objects(data_dir, ret_dict=True)

	runs_dir = 'runs/exp_s2b'

	p_env = p_Env(args.home_dir_data, gui=True, physics=False)
	
	for i, run_folder_dir in enumerate(glob.glob('{}/*{}'.format(runs_dir, args.exp_name))):
		# assert i == 0, run_folder_dir
		result_folder = run_folder_dir

		if args.eval_ct != -1:
			eval_file_dir_arr = glob.glob('{}/eval/*_ct_{}.json'.format(run_folder_dir, args.eval_ct))
		elif args.eval_epoch != -1:
			eval_file_dir_arr = glob.glob('{}/eval/*eval_epoch_{}_ct_*.json'.format(run_folder_dir, args.eval_epoch))
		assert len(eval_file_dir_arr) == 1, eval_file_dir_arr
		eval_file_dir = eval_file_dir_arr[0]

		eval_result_dict = load_json(eval_file_dir)

		for result_file_name in eval_result_dict:
			for i, one_result in enumerate(eval_result_dict[result_file_name]):
				print(result_file_name, i)
				pc_o = np.array(one_result['pc_o'])
				pc_h = np.array(one_result['pc_h'])
				gt_cp_score_o = np.array(one_result['gt_cp_score_o'])
				gt_cp_score_h = np.array(one_result['gt_cp_score_h'])
				pred_cp_score_o = np.array(one_result['pred_cp_score_o'])
				pred_cp_score_h = np.array(one_result['pred_cp_score_h'])

				pred_cp_top_k_idx_o = np.array(one_result['pred_cp_top_k_idx_o'])
				pred_cp_top_k_idx_h = np.array(one_result['pred_cp_top_k_idx_h'])

				loss_ce = one_result['loss_ce']
				loss_listnet = one_result['loss_listnet']

				if 'gt_cp_map_per_o' in one_result:
					gt_cp_map_per_o = np.array(one_result['gt_cp_map_per_o'])
					gt_cp_map_per_h = np.array(one_result['gt_cp_map_per_h'])
					_, gt_cp_top_k_idx_o = top_k_np(gt_cp_score_o, k=128)
					_, gt_cp_top_k_idx_h = top_k_np(gt_cp_score_h, k=128)
				
					gt_gt_cp_corr = create_gt_cp_corr_preload_discretize(gt_cp_map_per_o[np.newaxis, :], gt_cp_map_per_h[np.newaxis, :], gt_cp_top_k_idx_o[np.newaxis, :], gt_cp_top_k_idx_h[np.newaxis, :], n_gt_sample=args.n_gt_sample)
					gt_gt_cp_corr = gt_gt_cp_corr[0]

				gt_cp_corr = np.array(one_result['gt_cp_corr'])
				pred_cp_corr = np.array(one_result['pred_cp_corr'])
				pred_cp_corr_top_k_idx = np.array(one_result['pred_cp_corr_top_k_idx'])

				gt_pose = np.array(one_result['gt_pose'])
				pred_transl = np.array(one_result['pred_transl'])
				pred_aa = np.array(one_result['pred_aa'])
			
				# p_env.load_pair_w_pose(result_file_name, gt_pose[:3], gt_pose[3:], aa=True)
				p_env.load_pair_w_pose(result_file_name, pred_transl, pred_aa, aa=True)
				flag = input('in')
				if flag == 's':
					continue

				print('gt cp o {}'.format(print_helper(gt_cp_score_o)))
				print('gt cp h {}'.format(print_helper(gt_cp_score_h)))
				print('pred cp o {}'.format(print_helper(pred_cp_score_o)))
				print('pred cp h {}'.format(print_helper(pred_cp_score_h)))
				print('loss o', one_result['loss_o'], 'loss h', one_result['loss_h'])
				
				# print('calc o', np.mean(np.abs(pred_cp_score_o - gt_cp_score_o)**2), np.mean(np.abs(pred_cp_score_o/ np.max(pred_cp_score_o) - gt_cp_score_o)**2))
				# print('calc h', np.mean(np.abs(pred_cp_score_h - gt_cp_score_h)**2), np.mean(np.abs(pred_cp_score_h/ np.max(pred_cp_score_h) - gt_cp_score_h)**2))
				
				if flag != 'corr':
					
					
					# plot_pc_s(pc_o, gt_cp_score_o)
					# mayalab.show()
					plot_pc_s(pc_o, pred_cp_score_o, abs=False)
					mayalab.show()

					plot_pc(pc_o)
					plot_pc(pc_o[pred_cp_top_k_idx_o], color=[1, 0, 0])
					# plot_pc_s(pc_o, pred_cp_score_o, abs=False)
					mayalab.show()
					# plot_pc_s(pc_h, gt_cp_score_h)
					# mayalab.show()
					plot_pc_s(pc_h, pred_cp_score_h, abs=False)
					mayalab.show()

					#plot top k on hook
					plot_pc(pc_h)
					plot_pc(pc_h[pred_cp_top_k_idx_h], color=[1, 0, 0])
					# plot_pc_s(pc_h, pred_cp_score_h, abs=False)
					mayalab.show()

				
				#plot correspondence
				print('gt cp corr', np.mean(gt_cp_corr), np.min(gt_cp_corr), np.max(gt_cp_corr), np.std(gt_cp_corr))
				print('pred cp corr', np.mean(pred_cp_corr), np.min(pred_cp_corr), np.max(pred_cp_corr), np.std(pred_cp_corr))
				print('loss_ce {} loss_listnet {}'.format(loss_ce, loss_listnet))
				# if 'gt_cp_map_per_o' in one_result:
					# plot_corr(pc_o, pc_h, gt_pose[:3], gt_pose[3:], gt_cp_top_k_idx_o, gt_cp_top_k_idx_h, gt_gt_cp_corr, aa=True)
				plot_corr(pc_o, pc_h, pred_transl, pred_aa, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h, gt_cp_corr, aa=True)
				rot_center = plot_corr(pc_o, pc_h, pred_transl, pred_aa, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h, pred_cp_corr, aa=True)
				p_env.load_pair_w_pose(result_file_name, pred_transl + rot_center, pred_aa, aa=True)
				input('rw')