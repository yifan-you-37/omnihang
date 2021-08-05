import numpy as np
import os
import sys
import itertools
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from rotation_lib import *
def create_gt_cp_map(pc_o_idx, pc_h_idx, cp_map_o_dir, cp_map_h_dir, min_pose_idx):
	b_size = min_pose_idx.shape[0]
	gt_cp_score_o = np.zeros((b_size, 4096))
	gt_cp_score_h = np.zeros((b_size, 4096))
	non_nan_idx = []
	for ii in range(b_size):
		# print('min pose idx', min_pose_idx)
		# print(cp_map_o_dir[ii])
		cp_map_o_dir_tmp = cp_map_o_dir[ii][min_pose_idx[ii, 1]]
		cp_map_h_dir_tmp = cp_map_h_dir[ii][min_pose_idx[ii, 1]]

		cp_map_o_tmp = np.load(cp_map_o_dir_tmp)
		cp_map_h_tmp = np.load(cp_map_h_dir_tmp)

		cp_map_o_tmp = cp_map_o_tmp[pc_o_idx[ii]]
		cp_map_h_tmp = cp_map_h_tmp[pc_h_idx[ii]]

		gt_cp_score_o[ii] = cp_map_o_tmp / np.max(cp_map_o_tmp)
		gt_cp_score_h[ii] = cp_map_h_tmp / np.max(cp_map_h_tmp)
		if np.sum(np.isnan(cp_map_o_tmp)) == 0 and np.sum(np.isnan(cp_map_h_tmp)) == 0:
			non_nan_idx.append(ii)
	return gt_cp_score_o, gt_cp_score_h, np.array(non_nan_idx)

def load_gt_cp_per(pc_o_idx, pc_h_idx, cp_map_per_o_dir, cp_map_per_h_dir, min_pose_idx):
	b_size = min_pose_idx.shape[0]
	non_nan_idx = []

	all_cp_map_per_o = []
	all_cp_map_per_h = []
	for ii in range(b_size):
		cp_map_per_o_dir_tmp = cp_map_per_o_dir[ii][min_pose_idx[ii, 1]]
		cp_map_per_h_dir_tmp = cp_map_per_h_dir[ii][min_pose_idx[ii, 1]]

		cp_map_per_o_tmp = np.load(cp_map_per_o_dir_tmp)
		cp_map_per_h_tmp = np.load(cp_map_per_h_dir_tmp)

		cp_map_per_o_tmp = cp_map_per_o_tmp[:, pc_o_idx[ii]]
		cp_map_per_h_tmp = cp_map_per_h_tmp[:, pc_h_idx[ii]]	

		all_cp_map_per_o.append(cp_map_per_o_tmp)
		all_cp_map_per_h.append(cp_map_per_h_tmp)

		if np.sum(np.isnan(cp_map_per_o_tmp)) == 0 and np.sum(np.isnan(cp_map_per_h_tmp)) == 0:
			non_nan_idx.append(ii)
	return all_cp_map_per_o, all_cp_map_per_h, np.array(non_nan_idx)

def create_gt_cp_corr(pc_o_idx, pc_h_idx, cp_top_k_idx_o, cp_top_k_idx_h, cp_map_per_o_dir, cp_map_per_h_dir, min_pose_idx):
	b_size = min_pose_idx.shape[0]
	gt_cp_corr = np.zeros((b_size, 128, 128))
	non_nan_idx = []

	for ii in range(b_size):
		cp_map_per_o_dir_tmp = cp_map_per_o_dir[ii][min_pose_idx[ii, 1]]
		cp_map_per_h_dir_tmp = cp_map_per_h_dir[ii][min_pose_idx[ii, 1]]

		cp_map_per_o_tmp = np.load(cp_map_per_o_dir_tmp)
		cp_map_per_h_tmp = np.load(cp_map_per_h_dir_tmp)

		cp_map_per_o_tmp = cp_map_per_o_tmp[:, pc_o_idx[ii]]
		cp_map_per_h_tmp = cp_map_per_h_tmp[:, pc_h_idx[ii]]

		cp_map_per_o_tmp = cp_map_per_o_tmp[:, cp_top_k_idx_o[ii]]
		cp_map_per_h_tmp = cp_map_per_h_tmp[:, cp_top_k_idx_h[ii]]

		tmp_o = np.expand_dims(cp_map_per_o_tmp, axis=1)
		tmp_h = np.expand_dims(cp_map_per_h_tmp, axis=2)

		cp_corr_per = tmp_o * tmp_h
		cp_corr = np.sum(cp_corr_per, axis=0)
		
		gt_cp_corr[ii] = cp_corr / (np.max(cp_corr))
		
		if np.sum(np.isnan(cp_corr)) == 0:
			non_nan_idx.append(ii)
	
	return gt_cp_corr, np.array(non_nan_idx)

def create_gt_cp_corr_preload(all_cp_map_per_o, all_cp_map_per_h, cp_top_k_idx_o, cp_top_k_idx_h, discretize=False):
	b_size = cp_top_k_idx_o.shape[0]

	gt_cp_corr = np.zeros((b_size, 128, 128))
	if discretize:
		gt_cp_corr_discretize = np.zeros((b_size, 128 * 128), dtype=np.int32)
		
	non_nan_idx = []

	for ii in range(b_size):
		cp_map_per_o_tmp = all_cp_map_per_o[ii]
		cp_map_per_h_tmp = all_cp_map_per_h[ii]

		cp_map_per_o_tmp = cp_map_per_o_tmp[:, cp_top_k_idx_o[ii]]
		cp_map_per_h_tmp = cp_map_per_h_tmp[:, cp_top_k_idx_h[ii]]

		tmp_o = np.expand_dims(cp_map_per_o_tmp, axis=1)
		tmp_h = np.expand_dims(cp_map_per_h_tmp, axis=2)

		cp_corr_per = tmp_o * tmp_h
		cp_corr = np.sum(cp_corr_per, axis=0)

		gt_cp_corr[ii] = cp_corr / (np.max(cp_corr))
		if discretize:
			cp_corr_tmp = np.reshape(cp_corr, (-1))
			gt_cp_corr_top_k_idx = top_k_np(cp_corr_tmp, int(128 * 128 * 0.01)) 
			gt_cp_corr_discretize[ii][gt_cp_corr_top_k_idx] = 1
		
		if np.sum(np.isnan(cp_corr)) == 0:
			non_nan_idx.append(ii)
	
	if discretize:
		return gt_cp_corr, gt_cp_corr_discretize, np.array(non_nan_idx)
	else:
		return gt_cp_corr, np.array(non_nan_idx)



def create_gt_cp_corr_preload_discretize(all_cp_map_per_o, all_cp_map_per_h, cp_top_k_idx_o, cp_top_k_idx_h, n_gt_sample=128):
	b_size = cp_top_k_idx_o.shape[0]
	gt_cp_corr_discretize = np.zeros((b_size, 128, 128), dtype=np.int32)

	non_nan_idx = []

	n_zero_cp = 0
	n_cp = []
	cp_overlap_o = []
	cp_overlap_h = []
	for ii in range(b_size):
		cp_map_per_o_tmp = all_cp_map_per_o[ii]
		cp_map_per_h_tmp = all_cp_map_per_h[ii]
		
		cp_idx_arr = []
		for jj in range(all_cp_map_per_o[ii].shape[0]):
			_, gt_top_k_idx_o_tmp = top_k_np(all_cp_map_per_o[ii][jj], n_gt_sample)
			_, gt_top_k_idx_h_tmp = top_k_np(all_cp_map_per_h[ii][jj], n_gt_sample)
			
			_, _, tmp_idx_o = np.intersect1d(gt_top_k_idx_o_tmp, cp_top_k_idx_o[ii], return_indices=True)
			_, _, tmp_idx_h = np.intersect1d(gt_top_k_idx_h_tmp, cp_top_k_idx_h[ii], return_indices=True)
			
			cp_idx_arr += list(itertools.product(tmp_idx_o, tmp_idx_h))

			cp_overlap_o.append(1. * tmp_idx_o.size / (cp_top_k_idx_o[ii]).shape[0])
			cp_overlap_h.append(1. * tmp_idx_h.size / (cp_top_k_idx_h[ii]).shape[0])

		if len(cp_idx_arr) == 0:
			n_zero_cp += 1
		cp_idx_mat = idx_arr_to_mat(np.array(cp_idx_arr), cp_top_k_idx_o[ii].shape[0])
		n_cp.append(np.sum(cp_idx_mat))
		gt_cp_corr_discretize[ii] = cp_idx_mat
	
	n_cp = np.array(n_cp) * 1. / (gt_cp_corr_discretize.shape[1] * gt_cp_corr_discretize.shape[2])
	info_dict = {
		'zero_cp_prop': 1. * n_zero_cp / b_size,
		'n_cp_mean': np.mean(n_cp),
		'n_cp_std': np.std(n_cp),
		'n_cp_max': np.max(n_cp),
		'n_cp_min': np.min(n_cp),

		'cp_overlap_o': np.mean(cp_overlap_o),
		'cp_overlap_h': np.mean(cp_overlap_h)
		
	}
	return gt_cp_corr_discretize, info_dict

# def create_gt_cp_corr_preload_discretize_old(all_cp_map_per_o, all_cp_map_per_h, cp_top_k_idx_o, cp_top_k_idx_h):
# 	b_size = cp_top_k_idx_o.shape[0]
# 	gt_cp_corr_discretize = np.zeros((b_size, 128, 128), dtype=np.int32)

# 	non_nan_idx = []

# 	for ii in range(b_size):
# 		cp_map_per_o_tmp = all_cp_map_per_o[ii]
# 		cp_map_per_h_tmp = all_cp_map_per_h[ii]
		
# 		cp_idx_arr = []
# 		for jj in range(all_cp_map_per_o[ii].shape[0]):
# 			_, gt_top_k_idx_o_tmp = top_k_np(all_cp_map_per_o[ii][jj], 128)
# 			_, gt_top_k_idx_h_tmp = top_k_np(all_cp_map_per_h[ii][jj], 128)
# 			cp_idx_arr += list(itertools.product(gt_top_k_idx_o_tmp, gt_top_k_idx_h_tmp))

# 		cp_idx_mat = idx_arr_to_mat(np.array(cp_idx_arr), 4096, dtype=np.bool_)
# 		tmp_result = cp_idx_mat[cp_top_k_idx_o[ii]]
# 		gt_cp_corr_discretize[ii] = tmp_result[:, cp_top_k_idx_h[ii]]

# 	return gt_cp_corr_discretize

def pose_loss_l2(pc, pose_transl, pose_quat, all_pose_transl, all_pose_quat):
	# calculate loss between pose and all_pose using l2 pc loss
	pc_pose = transform_pc(pc, pose_transl, pose_quat)
	n_pose = all_pose_transl.shape[0]
	pc_all_pose = np.repeat(pc[np.newaxis, :, :], n_pose, axis=0)
	pc_all_pose = transform_pc_batch(pc_all_pose, all_pose_transl, all_pose_quat)

	pc_pose = np.expand_dims(pc_pose, 0)

	loss = np.sqrt(np.sum((pc_pose - pc_all_pose)**2, axis=-1))
	loss = np.mean(loss, axis=-1)

	return loss
	
def restore_model_s2a(epoch, save_top_dir, sess):
	ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
	variables = slim.get_variables_to_restore()
	variables_to_restore = [v for v in variables if 's2b_' not in v.name and (('s1_' in v.name) or ('s2a_' in v.name))]
	# for v in variables_to_restore:
		# print(v.name)
	saver = tf.train.Saver(variables_to_restore)
	print("restoring from %s" % ckpt_path)
	saver.restore(sess, ckpt_path)

def restore_model_s2b(epoch, save_top_dir, sess):
	ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
	variables = slim.get_variables_to_restore()
	variables_to_restore = [v for v in variables if 's3_' not in v.name and (('s1_' in v.name) or ('s2a_' in v.name) or ('s2b_' in v.name))]
	# for v in variables_to_restore:
		# print(v.name)
	saver = tf.train.Saver(variables_to_restore)
	print("restoring from %s" % ckpt_path)
	saver.restore(sess, ckpt_path)

def restore_model_s3_helper(ckpt_path, sess):
	variables = slim.get_variables_to_restore()
	variables_to_restore = [v for v in variables if 's3_' in v.name]
	saver = tf.train.Saver(variables_to_restore)
	saver.restore(sess, ckpt_path)
	print('restoring from %s' % ckpt_path)

def restore_model_s3(epoch, save_top_dir, sess):
	ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
	restore_model_s3_helper(ckpt_path, sess)

def restore_model_s3_second_last(save_top_dir, sess):
	second_last_ckpt_path = get_2nd_last_dir(save_top_dir, '*model.ckpt.index')

	if second_last_ckpt_path is None:
		return False
	second_last_ckpt_path = os.path.join(save_top_dir, second_last_ckpt_path)
	second_last_ckpt_path = second_last_ckpt_path[:-6]

	ckpt_path = second_last_ckpt_path
	restore_model_s3_helper(ckpt_path, sess)
	return True

