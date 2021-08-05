import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import random
# from contact_point_dataset_torch_multi_label import MyDataset 
from simple_dataset import MyDataset 
import os
import time
import argparse
from functools import partial

from torch.utils.data import DataLoader
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from train_helper import *
from bullet_helper import *
from rotation_lib import *
import s1_model_multi_label as s1_model
import s2a_model as s2a_model
import s2b_model_discretize as s2b_model
import s3_classifier_model as s3_model
from s2_utils import *

import s3_replay_buffer_pose as ReplayBuffer
import ES_multithread
import multiprocessing
from scipy.special import softmax

def calc_pose_cem_init(args, pc_o_all, pc_h_all, transl_s1_all, aa_s1_all, cp_top_k_idx_o_all, cp_top_k_idx_h_all, cp_corr_top_k_idx_o_all, cp_corr_top_k_idx_h_all):
	b_size = transl_s1_all.shape[0]
	cem_init_transl = np.zeros((b_size, 3))
	cem_init_aa = np.zeros((b_size, 3))
	cem_rotation_center_o = np.zeros((b_size, 3))
	pc_tmp_o_all = np.zeros((b_size, args.s3_num_cp, 3))
	pc_tmp_h_all = np.zeros((b_size, args.s3_num_cp, 3))
	pc_tmp_rotated_s1_o_all = np.zeros((b_size, args.s3_num_cp, 3))

	corr_idx_top_k_o_all = np.zeros((b_size, args.s3_num_cp))
	corr_idx_top_k_h_all = np.zeros((b_size, args.s3_num_cp))

	for bi in range(b_size):
		transl_s1 = transl_s1_all[bi]
		aa_s1 = aa_s1_all[bi]

		corr_idx_top_k_o = cp_top_k_idx_o_all[bi][cp_corr_top_k_idx_o_all[bi]]
		corr_idx_top_k_h = cp_top_k_idx_h_all[bi][cp_corr_top_k_idx_h_all[bi]]
		
		corr_idx_top_k_o = corr_idx_top_k_o[:args.s3_num_cp]
		corr_idx_top_k_h = corr_idx_top_k_h[:args.s3_num_cp]

		# use rotation from s1, and translation from s2b
		pc_tmp_o = pc_o_all[bi][corr_idx_top_k_o]
		pc_tmp_rotated_s1_o = transform_pc(pc_tmp_o, np.array([0, 0, 0]), aa_s1) 
		pc_tmp_h = pc_h_all[bi][corr_idx_top_k_h]

		ls_transl = np.mean(pc_tmp_h - pc_tmp_rotated_s1_o, axis=0)

		cem_init_transl[bi] = ls_transl
		cem_init_aa[bi] = aa_s1

		cem_rotation_center_o[bi] = np.mean(pc_tmp_h, axis=0)

		# debug info
		pc_tmp_o_all[bi] = pc_tmp_o
		pc_tmp_rotated_s1_o_all[bi] = pc_tmp_rotated_s1_o
		pc_tmp_h_all[bi] = pc_tmp_h
		corr_idx_top_k_o_all[bi] = corr_idx_top_k_o
		corr_idx_top_k_h_all[bi] = corr_idx_top_k_h
	
	info_dict = {
		's3_partial_pc_o': np.copy(pc_tmp_o_all),
		's3_partial_pc_rotated_s1_o': np.copy(pc_tmp_rotated_s1_o_all),
		's3_partial_pc_h': np.copy(pc_tmp_h_all),
		'cem_rotation_center_o': np.copy(cem_rotation_center_o),
		'corr_idx_top_k_o': np.copy(corr_idx_top_k_o_all),
		'corr_idx_top_k_h': np.copy(corr_idx_top_k_h_all)
	}
	return cem_init_transl, cem_init_aa, cem_rotation_center_o, info_dict

def cem_transform_pc_batch(pc_o, rotation_center_o, transl, aa):
	# TODO visualize
	if len(rotation_center_o.shape) == 1:
		rotation_center_o = rotation_center_o[np.newaxis, np.newaxis, :]
	if len(rotation_center_o.shape) == 2:
		rotation_center_o = np.expand_dims(rotation_center_o, axis=1)
	ret = transform_pc_batch(pc_o - rotation_center_o, transl, aa) + rotation_center_o
	return ret
	
def cem_eval(pose_np, pc_o, pc_h, rotation_center_o, sess_tf, pc_combined_pl, pred_succ_cla_score_tf):
	transl_np = pose_np[:, :, :3]
	aa_np = pose_np[:, :, 3:]
	b_size = transl_np.shape[0]
	pop_size = transl_np.shape[1]
	cem_score_all = np.zeros((b_size, pop_size))
	pc_combined_best_all = np.zeros((b_size, 4096 * 2, 4))
	for ii in range(b_size):
		transl_tmp = transl_np[ii]
		aa_tmp = aa_np[ii]
		
		pc_o_tmp = np.repeat(pc_o[ii][np.newaxis, :, :], pop_size, axis=0)
		pc_h_tmp = np.repeat(pc_h[ii][np.newaxis, :, :], pop_size, axis=0)
		rotation_center_o_tmp = rotation_center_o[ii][np.newaxis, np.newaxis, :]

		# transform object pc by subtracting rotation center first
		pc_o_tmp_transformed = cem_transform_pc_batch(pc_o_tmp, rotation_center_o_tmp, transl_tmp, aa_tmp)
		pc_combined_tmp = create_pc_combined_batch(pc_o_tmp_transformed, pc_h_tmp)

		pred_succ_cla_score = sess_tf.run(
			[pred_succ_cla_score_tf], 
			feed_dict={
				pc_combined_pl:pc_combined_tmp,
			}
		)
		# print(pred_succ_cla_score)
		pred_succ_cla_score = softmax(pred_succ_cla_score[0], axis=-1)
		# print('after', pred_succ_cla_score)
		cem_score_all[ii] = pred_succ_cla_score[:, 1]
		
		max_idx = np.argmax(pred_succ_cla_score[:, 1])
		pc_combined_best_all[ii] = pc_combined_tmp[max_idx]
		# print(pred_succ_cla_score)
		# for jj in range(pred_succ_cla_score.shape[0]):
		# 	if pred_succ_cla_score[jj, 1] > 0.9:
		# 		np.save('tmp/pc_combined_{}_{}.npy'.format(jj, pred_succ_cla_score[jj, 1]), pc_combined_tmp[jj])
			# if pred_succ_cla_score[jj, 1] < 0.1:
				# np.save('tmp/pc_combined_{}_{}.npy'.format(jj, pred_succ_cla_score[jj, 1]), pc_combined_tmp[jj])

	# pred_succ_best = sess_tf.run(
	# 	[pred_succ_cla_score_tf],
	# 	feed_dict={
	# 		pc_combined_pl: pc_combined_best_all
	# 	}
	# )
	# print('best all score', softmax(pred_succ_best, axis=-1))
	return cem_score_all, pc_combined_best_all

import s3_bullet_checker as bullet_checker
import s3_bullet_checker_eval as bullet_checker_eval

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


def train(args, train_set, train_loader, test_set, test_loader, writer, result_folder, file_name, extra_dict=None):
	model_folder = os.path.join(result_folder, 'models')
	can_write = not (writer is None)

	# stage 1
	pc_o_pl, pc_h_pl, z_pl, gt_transl_pl, gt_aa_pl, pose_mult_pl = s1_model.placeholder_inputs(args.batch_size, 4096, args)
	pred_transl_tf, pred_aa_tf, end_points_s1 = s1_model.get_model(pc_o_pl, pc_h_pl, z_pl)
	loss_transl_tf, loss_aa_tf, min_pose_idx_tf = s1_model.get_loss(pred_transl_tf, pred_aa_tf, gt_transl_pl, gt_aa_pl, pose_mult_pl, float(args.loss_transl_const), end_points_s1)
	loss_s1_tf = float(args.loss_transl_const) * loss_transl_tf + loss_aa_tf
	
	# stage 2a
	pc_combined_pl, gt_cp_score_o_pl, gt_cp_score_h_pl, non_nan_mask_pl = s2a_model.placeholder_inputs(args.batch_size, 4096, args)
	pred_cp_score_o_tf, pred_cp_score_h_tf, end_points_s2a = s2a_model.get_model(pc_combined_pl, 4096)
	loss_s2a_o_tf, loss_s2a_h_tf = s2a_model.get_loss(pred_cp_score_o_tf, pred_cp_score_h_tf, gt_cp_score_o_pl, gt_cp_score_h_pl, end_points_s2a)

	# stage 2b
	gt_cp_corr_pl = s2b_model.placeholder_inputs(args.batch_size, 4096, args)
	pred_cp_corr_tf, end_points = s2b_model.get_model(pc_combined_pl, pred_cp_score_o_tf, pred_cp_score_h_tf, 4096, args.batch_size)

	pred_cp_top_k_idx_o_tf = end_points['pred_cp_top_k_idx_o']
	pred_cp_top_k_idx_h_tf = end_points['pred_cp_top_k_idx_h']

	pred_cp_corr_logit_tf = end_points['pred_cp_corr_logit']
	_, pred_cp_corr_top_k_idx_tf = tf.nn.top_k(pred_cp_corr_logit_tf[:, :, 1], k=args.top_k_corr, sorted=True) 
	pred_cp_corr_top_k_idx_o_tf = pred_cp_corr_top_k_idx_tf // args.top_k_h
	pred_cp_corr_top_k_idx_h_tf = pred_cp_corr_top_k_idx_tf % args.top_k_o

	loss_listnet_tf, loss_ce_tf = s2b_model.get_loss(pred_cp_corr_logit_tf, gt_cp_corr_pl, end_points)

	#stage 3
	_, gt_succ_label_pl = s3_model.placeholder_inputs(args.batch_size, 4096, with_normal=False, args=args)
	pred_succ_cla_score_tf, end_points = s3_model.get_model(pc_combined_pl)

	pred_succ_cla_tf = tf.math.argmax(pred_succ_cla_score_tf, axis=-1)
	# loss_succ_cla_tf = s3_model.get_loss(pred_succ_cla_score_tf, gt_succ_label_pl, end_points)

	# loss_tf = loss_listnet_tf + loss_ce_tf
	# loss_tf = tf.boolean_mask(loss_tf, non_nan_mask_pl)
	# loss_tf = tf.reduce_mean(loss_tf)
	# print('loss tf', loss_tf)
	# train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_tf)

	init_op = tf.group(tf.global_variables_initializer(),
				tf.local_variables_initializer())

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.Session(config=config)
	sess.run(init_op)
	saver = tf.train.Saver(max_to_keep=1000)
	loss_tracker = LossTracker()
	loss_tracker_test = LossTracker()

	epoch_init = 0
	if not train_loader is None:
		epoch_iter = len(train_loader)
	
	if args.pretrain_s2b:
		pretrain_s2b_folder = os.path.abspath(os.path.join(result_folder, '..', '..', args.pretrain_s2b_folder, args.pretrain_s2b_model_name, 'models'))
		restore_model_s2b(args.pretrain_s2b_epoch, pretrain_s2b_folder, sess)

	if args.pretrain_s3:
		pretrain_s3_folder = os.path.abspath(os.path.join(result_folder, '..', '..', args.pretrain_s3_folder, args.pretrain_s3_model_name, 'models'))

		if args.pretrain_s3_folder_dir != '':
			pretrain_s3_folder = args.pretrain_s3_folder_dir
		restore_model_s3(args.pretrain_s3_epoch, pretrain_s3_folder, sess)
	else:
		print(args.s3_model_dir)
		tmp = restore_model_s3_second_last(args.s3_model_dir, sess)
		assert tmp # make sure that one model is restored

	if args.restore_model_epoch != -1:
		epoch_init = args.restore_model_epoch
		restore_model_folder = os.path.abspath(os.path.join(result_folder, '..', args.restore_model_name, 'models'))
		restore_model_generic(epoch_init, restore_model_folder, saver, sess)

	total_ct = 0

	cem = ES_multithread.Searcher(
        action_dim=6,
        max_action=float(args.cem_max_transl),
        max_action_aa=float(args.cem_max_aa),
        sigma_init=float(args.cem_sigma_init_transl),
        sigma_init_aa=float(args.cem_sigma_init_aa),
        pop_size=args.cem_pop_size,
        damp=float(args.cem_damp_transl),
        damp_limit=float(args.cem_damp_limit_transl),
        damp_aa=float(args.cem_damp_aa),
        damp_limit_aa=float(args.cem_damp_limit_aa),
        parents=args.cem_parents,		
	)
	
	fcl_hook_dict = extra_dict['fcl_hook_dict']
	fcl_object_dict = extra_dict['fcl_object_dict']

	if not args.no_bullet_check:
		pool = multiprocessing.Pool(processes=args.batch_size)
		p_list = pool.map(partial(p_init_multithread, gui=args.bullet_gui), range(args.batch_size))

	if not args.run_test:
		saved_buffer_name_list = [get_2nd_last_dir(args.s3_buffer_dir)]
		replay_buffer = ReplayBuffer.combine_replay_buffer(saved_buffer_name_list, train_set, args.home_dir_data, preload_data=False)

	eval_folder_dir = os.path.join(result_folder, 'eval')
	mkdir_if_not(eval_folder_dir)
	# for epoch_i in range(args.max_epochs):
	epoch_i = 0
	tested = False
	while True:
		if epoch_i == args.max_epochs:
			break
		loss_tracker.reset()

		run_test = False
		if (not args.no_eval) and (((epoch_i + 1) % args.eval_epoch_freq == 0) or args.run_test):
			if not tested:
				run_test = True
				epoch_i -= 1
				tested = True
			if tested:
				tested = False
		if args.run_test:
			epoch_i = 0
		if not run_test:
			loader = train_loader
			dataset = train_set
		else:
			loader = test_loader
			dataset = test_set
		
		info_dict_all = {}

		for i, batch_dict in enumerate(loader):
			total_ct += 1

			if not args.no_bullet_check:
				if (total_ct + 1) % 20 == 0:
					print('reset')
					p_list = pool.map(partial(p_reset_multithread, p_list=p_list, gui=args.bullet_gui), range(args.batch_size))
				
			log_it = ((total_ct % args.log_freq ) == 0) and can_write
			pc_o = batch_dict['input1']
			pc_h = batch_dict['input2']
			b_size = pc_o.shape[0]
			object_urdf = batch_dict['urdf_o'] 
			hook_urdf = batch_dict['urdf_h']   
			result_file_name = batch_dict['result_file_name']

			fcl_hook_model = [fcl_hook_dict[name] for name in batch_dict['hook_name']]
			fcl_object_model = [fcl_object_dict[name] for name in batch_dict['object_name']]
			
			if run_test:
				eval_sample_n = args.eval_sample_n
			else:
				eval_sample_n = 1 
			info_dict = {}
			bullet_succ = np.zeros((b_size, eval_sample_n))
			for i_eval in range(args.eval_sample_n):
				z = np.random.normal(size=(pc_o.shape[0],1,32))
				feed_dict_s1 = {
					pc_o_pl: pc_o[:, :, :3],
					pc_h_pl: pc_h[:, :, :3],
					z_pl: z,
				}
				
				pred_transl, pred_aa = sess.run([
					pred_transl_tf, pred_aa_tf
				], feed_dict=feed_dict_s1)

				# stage 2a
				pc_combined = create_pc_combined_batch(pc_o, pc_h, pred_transl, pred_aa, aa=True)

				non_nan_mask = np.ones((b_size), dtype=np.bool)

				feed_dict_s2a = {
					pc_combined_pl: pc_combined,
					non_nan_mask_pl: non_nan_mask
				}

				pred_cp_score_o, pred_cp_score_h, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h = sess.run([
					pred_cp_score_o_tf, pred_cp_score_h_tf, pred_cp_top_k_idx_o_tf, pred_cp_top_k_idx_h_tf
				], feed_dict=feed_dict_s2a)

				# stage 2b
				feed_dict = {
					pc_combined_pl: pc_combined,
					non_nan_mask_pl: non_nan_mask
				}

				pred_cp_corr, pred_cp_corr_top_k_idx, pred_cp_corr_top_k_idx_o, pred_cp_corr_top_k_idx_h = sess.run([
					pred_cp_corr_tf, pred_cp_corr_top_k_idx_tf, pred_cp_corr_top_k_idx_o_tf, pred_cp_corr_top_k_idx_h_tf
				], feed_dict=feed_dict)

				# stage 3
				cem_init_transl, cem_init_aa, cem_rotation_center_o, cem_info_dict = calc_pose_cem_init(
					args=args,
					pc_o_all=pc_o[:, :, :3],
					pc_h_all=pc_h[:, :, :3],
					transl_s1_all=pred_transl,
					aa_s1_all=pred_aa,
					cp_top_k_idx_o_all=pred_cp_top_k_idx_o,
					cp_top_k_idx_h_all= pred_cp_top_k_idx_h,
					cp_corr_top_k_idx_o_all=pred_cp_corr_top_k_idx_o,
					cp_corr_top_k_idx_h_all=pred_cp_corr_top_k_idx_h,
				)

				pc_o_cem_init = transform_pc_batch(pc_o[:, :, :3], cem_init_transl, cem_init_aa)
				
				cem_eval_partial = partial(cem_eval,
					pc_o=pc_o_cem_init,
					pc_h=pc_h[:, :, :3],
					rotation_center_o=cem_rotation_center_o,
					sess_tf=sess,
					pc_combined_pl=pc_combined_pl,
					pred_succ_cla_score_tf=pred_succ_cla_score_tf,
				)

				cem_search_info_dict = {
					'cem_elite_pose': np.zeros((b_size, args.cem_n_iter, 6)),
					'cem_elite_pose_scores': np.zeros((b_size, args.cem_n_iter))
				}
				cem_max_score = np.zeros((b_size))
				cem_out_transl = np.zeros((b_size, 3))
				cem_out_aa = np.zeros((b_size, 3))
				cem_out_pose = np.zeros((b_size, 6))
				for ii in range(args.cem_run_n):
					_, cem_out_pose_tmp, cem_score_tmp, cem_search_info_dict_tmp = cem.search(
						b_size,
						np.array([[0, 0, 0, 1e-6, 0, 0]] * b_size), 
						cem_eval_partial,
						n_iter=args.cem_n_iter,
						elitism=True,
						visualize=False,
						visualize_func=None,
					)
					for jj in range(b_size):
						cur_score = cem_score_tmp[jj]
						if cem_max_score[jj] < cur_score:
							cem_max_score[jj] = cur_score
							cem_out_transl[jj] = cem_out_pose_tmp[jj, :3]
							cem_out_aa[jj] = cem_out_pose_tmp[jj, 3:]
							cem_out_pose[jj] = cem_out_pose_tmp[jj]
							cem_search_info_dict['cem_elite_pose'][jj] = cem_search_info_dict_tmp['cem_elite_pose'][jj]
							cem_search_info_dict['cem_elite_pose_scores'][jj] = cem_search_info_dict_tmp['cem_elite_pose_scores'][jj]
				# tmp, _ = cem_eval_partial(np.append(cem_out_transl[:, np.newaxis, :], cem_out_aa[:, np.newaxis, :], axis=-1))
				# print('verify', np.max(np.abs(tmp[:, 0] - cem_max_score)))
				# convert the cem output pose to object pose
				pc_o_cem = cem_transform_pc_batch(pc_o_cem_init[:, :, :3], cem_rotation_center_o, cem_out_transl, cem_out_aa)
				
				final_pred_transl, final_pred_aa = best_fit_transform_batch(pc_o[:, :, :3], pc_o_cem)

				if not args.no_bullet_check:
					bullet_check_one_pose = bullet_checker_eval.check_one_pose_simple if args.use_bullet_checker else bullet_checker.check_one_pose_simple 
					# bullet check
					bullet_check_func = partial(
						bullet_check,
						bullet_check_one_pose=bullet_check_one_pose,
						transl=final_pred_transl,
						aa=final_pred_aa,
						p_list=p_list,
						result_file_name=result_file_name,
						hook_urdf=hook_urdf,
						object_urdf=object_urdf,
						fcl_hook_model=fcl_hook_model,
						fcl_object_model=fcl_object_model,
						gui=args.bullet_gui,
					)

					for bi, (flag_tmp, bullet_final_transl, bullet_final_quat) in enumerate(pool.imap(bullet_check_func, range(b_size))):
						succ = 1. if flag_tmp else 0.
						bullet_succ[bi, i_eval] = succ

						hook_name, object_name = split_result_file_name(result_file_name[bi])
						if not run_test:
							replay_buffer.add(object_name, hook_name, np.append(final_pred_transl[bi], final_pred_aa[bi]), succ)

				for ii in range(b_size):
					if i_eval == 0:
						info_dict[result_file_name[ii]] = []

					info_dict[result_file_name[ii]].append({
						'z': z[ii].tolist(),
						's1_transl': pred_transl[ii].tolist(),
						's1_aa': pred_aa[ii].tolist(),
						'cem_init_transl': cem_init_transl[ii].tolist(),
						'cem_init_aa': cem_init_aa[ii].tolist(),
					
						'succ': bullet_succ[ii, i_eval],
						'final_pred_transl': final_pred_transl[ii].tolist(),
						'final_pred_aa': final_pred_aa[ii].tolist(),
						'cem_out_transl': cem_out_pose[ii, :3].tolist(),
						'cem_out_aa': cem_out_pose[ii, 3:].tolist(),
						# 'cem_elite_pose': cem_elite_pose[ii].tolist()
					})
					for tmp_key in cem_info_dict: 
						info_dict[result_file_name[ii]][-1][tmp_key] = cem_info_dict[tmp_key][ii].tolist()

					for tmp_key in cem_search_info_dict: 
						info_dict[result_file_name[ii]][-1][tmp_key] = cem_search_info_dict[tmp_key][ii].tolist()

			info_dict_all.update(info_dict)
			
			loss_dict = {
				'bullet_succ_acc': np.mean(bullet_succ),
				'bullet_succ_acc_max': np.mean(np.max(bullet_succ, axis=-1))
			}
			loss_tracker.add_dict(loss_dict)

			if log_it:
				write_tb(loss_dict, writer, 'test' if run_test else 'train', total_ct)
			print('epoch {} iter {}/{} {}'.format(epoch_i, i, epoch_iter, loss_dict_to_str(loss_dict)))
			loss_dict_epoch = loss_tracker.stat()
			print('cumulative, epoch {} {} {}'.format(epoch_i, 'test' if run_test else 'train', loss_dict_to_str(loss_dict_epoch)))

			# periodically save buffer
			if int(total_ct) % args.s3_buffer_freq == 0 and (not run_test):
				print('save buffer', args.s3_buffer_dir, replay_buffer.buffer_succ.size, replay_buffer.buffer_fail.size, )
				replay_buffer.save_pkl(args.s3_buffer_dir, replay_buffer)

				# save info dict
				info_dict_dir = os.path.join(eval_folder_dir, '{}_eval_epoch_{}_ct_{}_{}.json'.format(file_name, str(epoch_i + 1), int(total_ct), 'test' if run_test else 'train'))
				save_json(info_dict_dir, info_dict_all)
				
			
			# periodically load s3 model
			if int(total_ct) % args.s3_model_freq == 0 and (not run_test):
				restore_model_s3_second_last(args.s3_model_dir, sess)

			if (total_ct % args.model_save_freq == 0) and not args.no_save and (not run_test):
				save_model_generic(epoch_init + total_ct, model_folder, saver, sess)

			if total_ct % 5 == 0 or total_ct == epoch_iter:
				# save info dict
				info_dict_dir = os.path.join(eval_folder_dir, '{}_eval_epoch_{}_{}.json'.format(file_name, str(epoch_i + 1), 'test' if run_test else 'train'))
				save_json(info_dict_dir, info_dict_all)

		loss_dict_epoch = loss_tracker.stat()
		if can_write:
			write_tb(loss_dict_epoch, writer, 'test_epoch' if run_test else 'train_epoch', total_ct)
		print('epoch {} {} {}'.format(epoch_i, 'test' if run_test else 'train', loss_dict_to_str(loss_dict_epoch)))
		
		epoch_i += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")

	parser.add_argument('--model_name', default='s3_rl_collect')
	parser.add_argument('--comment', default='')
	parser.add_argument('--exp_name', default='exp_s3')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--log_freq', type=int, default=2)


	parser.add_argument('--train_list', default='train_list')
	parser.add_argument('--test_list', default='test_list')
	parser.add_argument('--restrict_object_cat', default='')

	parser.add_argument('--run_test', action='store_true')
	parser.add_argument('--no_save', action='store_true')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--restore_model_name', default='')
	parser.add_argument('--restore_model_epoch', type=int, default=-1)
	parser.add_argument('--max_epochs', type=int, default=10000)
	parser.add_argument('--eval_epoch_freq', type=int, default=2)
	parser.add_argument('--eval_sample_n', type=int, default=1)
	parser.add_argument('--model_save_freq', type=int, default=3000)
	parser.add_argument('--no_eval', action='store_true')

	parser.add_argument('--loss_transl_const', default=1)

	parser.add_argument('--data_one_pose', action='store_true')
	parser.add_argument('--data_vary_scale', action='store_true')
	parser.add_argument('--data_more_pose', action='store_true')
	parser.add_argument('--data_vary_scale_more_pose', action='store_true')

	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--learning_rate', type=float, default=1e-4)

	#s1 argument
	parser.add_argument('--z_dim', type=int, default=32)

	#s2 argument
	parser.add_argument('--top_k_o', type=int, default=128)
	parser.add_argument('--top_k_h', type=int, default=128)
	parser.add_argument('--n_gt_sample', type=int, default=128)
	parser.add_argument('--top_k_corr', type=int, default=256)
	parser.add_argument('--pose_loss_l2', action='store_true')

	#s3 argument
	parser.add_argument('--s3_num_cp', type=int, default=3)
	parser.add_argument('--cem_run_n', type=int, default=1)
	parser.add_argument('--cem_n_iter', type=int, default=10)
	parser.add_argument('--cem_max_transl', default=0.02)
	parser.add_argument('--cem_max_aa', default=0.5)
	parser.add_argument('--cem_sigma_init_transl', default=1e-2)
	parser.add_argument('--cem_sigma_init_aa', default=1e-1)
	parser.add_argument('--cem_pop_size', type=int, default=32)
	parser.add_argument('--cem_damp_transl', default=0.005)
	parser.add_argument('--cem_damp_limit_transl', default=1e-2)
	parser.add_argument('--cem_damp_aa', default=0.1)
	parser.add_argument('--cem_damp_limit_aa', default=0.1)
	parser.add_argument('--cem_parents', type=int, default=10)

	parser.add_argument('--bullet_gui', action='store_true')
	parser.add_argument('--s3_train_folder_dir', default='/juno/downloads/new_hang_training/')
	parser.add_argument('--s3_train_name', default='s3')
	parser.add_argument('--s3_device_name', default='bohg4')
	parser.add_argument('--s3_buffer_dir', default='')
	parser.add_argument('--s3_model_dir', default='')
	parser.add_argument('--no_fcl', action='store_true')
	parser.add_argument('--s3_buffer_freq', default=1000, type=int)
	parser.add_argument('--s3_model_freq', default=1000, type=int)
	parser.add_argument('--use_bullet_checker', action='store_true')


	parser.add_argument('--pretrain_s2b', action='store_true')
	parser.add_argument('--pretrain_s2b_folder', default='exp_s2b')
	parser.add_argument('--pretrain_s2b_model_name', default='Mar17_14-14-19_s2b_discretize_model')
	parser.add_argument('--pretrain_s2b_epoch', default=57000, type=int)

	#parser.add_argument('--pretrain_s2b_model_name', default='Feb10_00-31-36_s2b_discretize_model_fixed')
	#parser.add_argument('--pretrain_s2b_epoch', default=93000, type=int)

	parser.add_argument('--pretrain_s3', action='store_true')
	parser.add_argument('--pretrain_s3_folder', default='exp_s3')
	parser.add_argument('--pretrain_s3_folder_dir', default='')
	parser.add_argument('--pretrain_s3_model_name', default='Mar19_03-43-24_s3_rl_train_all')
	parser.add_argument('--pretrain_s3_epoch', default=18000, type=int)

	parser.add_argument('--lin', action='store_true')
	parser.add_argument('--no_bullet_check', action='store_true')
	parser.add_argument('--seed', type=int, default=134)

	parser.add_argument('--parallel_n', default=-1, type=int)
	parser.add_argument('--parallel_id', default=-1, type=int)

	args = parser.parse_args()
	args.home_dir_data = os.path.abspath(args.home_dir_data)

	args.data_more_pose = True

	print("args.seed",args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	if args.no_bullet_check:
		args.no_fcl = True


	file_name = "{}".format(args.model_name)
	file_name += '_{}'.format(args.restrict_object_cat) if args.restrict_object_cat != '' else ''
	file_name += "_{}".format(args.comment) if args.comment != "" else ""
	file_name += '_overfit' if args.overfit else ''
	folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name

	if args.use_bullet_checker:
		args.no_fcl = True

	if args.run_test:
		assert args.pretrain_s2b
		folder_name += '_test'

	result_folder = 'runs/{}'.format(folder_name) 
	if args.exp_name is not "":
		result_folder = 'runs/{}/{}'.format(args.exp_name, folder_name)
	if args.debug: 
		result_folder = 'runs/debug/{}'.format(folder_name)

	model_folder = os.path.join(result_folder, 'models')

	if not os.path.exists(model_folder):
		os.makedirs(result_folder)

	if args.debug:
		args.s3_train_name = 'debug'
	else:
		args.s3_train_name += "_{}".format(args.comment) if args.comment != "" else ""
	if not args.run_test:
		s3_train_dir = os.path.join(args.s3_train_folder_dir, args.s3_train_name)
		mkdir_if_not(s3_train_dir)

		args.s3_buffer_dir = os.path.join(s3_train_dir, 'buffers')
		mkdir_if_not(args.s3_buffer_dir)
		args.s3_buffer_dir = os.path.join(args.s3_buffer_dir, args.s3_device_name)
		mkdir_if_not(args.s3_buffer_dir)
		
		args.s3_model_dir = os.path.join(s3_train_dir, 'models')
		mkdir_if_not(args.s3_model_dir)
		

	print("---------------------------------------")
	print("Model Name: {}, Train List: {}, Test List: {}".format(args.model_name, args.train_list, args.test_list))
	print("---------------------------------------")

	if args.run_test:
		print("Restore Model: {}, Test Sample N: {}".format(args.restore_model_name, args.eval_sample_n))
		print("---------------------------------------")
	
	writer = None
	if not args.run_test:
		writer = SummaryWriter(log_dir=result_folder, comment=file_name)

	#record all parameters value
	with open("{}/parameters.txt".format(result_folder), 'w') as file:
		for key in sorted(vars(args).keys()):
			value = vars(args)[key]
			file.write("{} = {}\n".format(key, value))

	cp_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp')
	train_list_dir = os.path.join(cp_result_folder_dir, 'labels', '{}.txt'.format(args.train_list))
	test_list_dir = os.path.join(cp_result_folder_dir, 'labels', '{}.txt'.format(args.test_list))

	print('TRAIN_LIST:', args.train_list, train_list_dir)
	print('TEST_LIST:', args.test_list, test_list_dir)

	# if args.overfit:
		# args.no_eval = True
		# args.no_save = True
	if args.run_test:
		args.max_epochs = 1

	if args.restore_model_name != '':
		assert args.restore_model_epoch != -1

	if args.restore_model_epoch != -1:
		assert args.restore_model_name != ''

	train_loader = None
	if (not args.run_test) or args.overfit:
		train_set = MyDataset(args.home_dir_data, train_list_dir, is_train=True, use_partial_pc=True, use_fcl=(not args.no_fcl), split_n=args.parallel_n, split_id=args.parallel_id, args=args)
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
							num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict, drop_last=True)
	
	if not args.no_eval:
		test_set = MyDataset(args.home_dir_data, test_list_dir, is_train=False, use_partial_pc=True, use_fcl=(not args.no_fcl), split_n=args.parallel_n, split_id=args.parallel_id, args=args)

		test_loader = DataLoader(train_set if args.overfit else test_set, batch_size=args.batch_size, shuffle=True,
									num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict, drop_last=True)
	else:
		test_set = None
		test_loader = None
	# if not args.run_test:
	# 	print('len of train {} len of test {}'.format(len(train_set), len(test_set)))
	# else:
	# 	print('len of train {} len of test {}'.format(len(test_set), len(test_set)))
	if args.run_test:
		train_set = test_set
		train_loader = test_loader
	extra_dict = {
		'fcl_object_dict': train_set.fcl_object_dict,
		'fcl_hook_dict': train_set.fcl_hook_dict,
	}
	if not args.run_test:
		train(args, train_set, train_loader, test_set, test_loader, writer, result_folder, file_name, extra_dict=extra_dict)
	else:
		train(args, test_set, test_loader, test_set, test_loader, writer, result_folder, file_name, extra_dict=extra_dict)


