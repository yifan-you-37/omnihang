import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import random
# from contact_point_dataset_torch_multi_label import MyDataset 
from hang_dataset import MyDataset 
import os
import time
import argparse

from torch.utils.data import DataLoader
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import datetime
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from train_helper import *
from rotation_lib import *
import s1_model as s1_model
import s2a_model as s2a_model
import s2b_model as my_model

from s2_utils import *

def train(args, train_loader, test_loader, writer, result_folder, file_name):
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
	gt_cp_corr_pl = my_model.placeholder_inputs(args.batch_size, 4096, args)
	pred_cp_corr_tf, end_points = my_model.get_model(pc_combined_pl, pred_cp_score_o_tf, pred_cp_score_h_tf, 4096, args.batch_size)

	pred_cp_top_k_idx_o_tf = end_points['pred_cp_top_k_idx_o']
	pred_cp_top_k_idx_h_tf = end_points['pred_cp_top_k_idx_h']

	pred_cp_corr_logit_tf = end_points['pred_cp_corr_logit']
	_, pred_cp_corr_top_k_idx_tf = tf.nn.top_k(pred_cp_corr_logit_tf[:, :, 1], k=256, sorted=True) 

	loss_listnet_tf, loss_ce_tf = my_model.get_loss(pred_cp_corr_logit_tf, gt_cp_corr_pl, end_points)

	loss_tf = loss_listnet_tf + loss_ce_tf
	loss_tf = tf.boolean_mask(loss_tf, non_nan_mask_pl)
	loss_tf = tf.reduce_mean(loss_tf)
	print('loss tf', loss_tf)
	train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_tf)

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
	
	if args.pretrain_s2a:
		pretrain_s2a_folder = os.path.abspath(os.path.join(result_folder, '..', '..', args.pretrain_s2a_folder, args.pretrain_s2a_model_name, 'models'))
		restore_model_s2a(args.pretrain_s2a_epoch, pretrain_s2a_folder, sess)

	if args.restore_model_epoch != -1:
		epoch_init = args.restore_model_epoch
		restore_model_folder = os.path.abspath(os.path.join(result_folder, '..', args.restore_model_name, 'models'))
		restore_model_generic(epoch_init, restore_model_folder, saver, sess)

	total_ct = 0

	for epoch_i in range(args.max_epochs):
		loss_tracker.reset()
		for i, batch_dict in enumerate(train_loader):
			if args.run_test:
				break
			total_ct += 1
			log_it = ((total_ct % args.log_freq ) == 0) and can_write
			pc_o = batch_dict['input1']
			pc_h = batch_dict['input2']
			gt_pose = batch_dict['output4']
			n_pose = batch_dict['n_pose']
			b_size = gt_pose.shape[0]

			pc_o_idx = batch_dict['pc_o_idx']
			pc_h_idx = batch_dict['pc_h_idx']
			cp_map_o_dir = batch_dict['cp_map_o_dir']
			cp_map_h_dir = batch_dict['cp_map_h_dir']
			cp_map_per_o_dir = batch_dict['cp_map_per_o_dir']
			cp_map_per_h_dir = batch_dict['cp_map_per_h_dir']

			pose_mult = np.ones((b_size, gt_pose.shape[1]))
			for ii in range(b_size):
				pose_mult[ii][n_pose[ii]:] = np.inf
			feed_dict_s1 = {
				pc_o_pl: pc_o[:, :, :3],
				pc_h_pl: pc_h[:, :, :3],
				gt_transl_pl: gt_pose[:, :, :3],
				gt_aa_pl: gt_pose[:, :, 3:],
				z_pl: np.random.normal(size=(pc_o.shape[0],1,32)),
				pose_mult_pl: pose_mult,
			}
			
			pred_transl, pred_aa, loss_transl_val, loss_aa_val, min_pose_idx = sess.run([
				pred_transl_tf, pred_aa_tf, loss_transl_tf, loss_aa_tf, min_pose_idx_tf
			], feed_dict=feed_dict_s1)
			gt_min_pose = np.squeeze(np.take_along_axis(gt_pose, np.expand_dims(min_pose_idx[:, 1:], axis=1), axis=1), axis=1)
			angle_diff = np.mean(angle_diff_batch(gt_min_pose[:, 3:], pred_aa, aa=True, degree=True))

			# stage 2a
			pc_combined = create_pc_combined_batch(pc_o, pc_h, pred_transl, pred_aa, aa=True)

			min_pose_idx_s2 = np.ones((b_size, 2), dtype=np.int32) * -1

			min_pose_idx_s2 = min_pose_idx
			gt_min_pose_s2 = np.squeeze(np.take_along_axis(gt_pose, np.expand_dims(min_pose_idx_s2[:, 1:], axis=1), axis=1), axis=1)
			angle_diff_min_pose_s2 = np.mean(angle_diff_batch(gt_min_pose_s2[:, 3:], pred_aa, aa=True, degree=True))

			gt_cp_score_o, gt_cp_score_h, non_nan_idx = create_gt_cp_map(pc_o_idx, pc_h_idx, cp_map_o_dir, cp_map_h_dir, min_pose_idx_s2)
			gt_cp_map_per_o, gt_cp_map_per_h, non_nan_idx_per = load_gt_cp_per(pc_o_idx, pc_h_idx, cp_map_per_o_dir, cp_map_per_h_dir, min_pose_idx_s2)

			non_nan_idx = np.intersect1d(non_nan_idx, non_nan_idx_per)

			non_nan_mask = np.zeros((b_size), dtype=np.bool)
			non_nan_mask[non_nan_idx] = True
			if len(non_nan_idx) == 0:
				continue

			feed_dict_s2a = {
				pc_combined_pl: pc_combined,
				gt_cp_score_o_pl: gt_cp_score_o,
				gt_cp_score_h_pl: gt_cp_score_h,
				non_nan_mask_pl: non_nan_mask
			}

			pred_cp_score_o, pred_cp_score_h, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h, loss_o_val, loss_h_val = sess.run([
				pred_cp_score_o_tf, pred_cp_score_h_tf, pred_cp_top_k_idx_o_tf, pred_cp_top_k_idx_h_tf, loss_s2a_o_tf, loss_s2a_h_tf
			], feed_dict=feed_dict_s2a)

			# stage 2b
			gt_cp_corr, cp_info_dict = create_gt_cp_corr_preload_discretize(gt_cp_map_per_o, gt_cp_map_per_h, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h, n_gt_sample=args.n_gt_sample)
			gt_cp_corr = gt_cp_corr.reshape((b_size, -1))

			feed_dict = {
				pc_combined_pl: pc_combined,
				gt_cp_corr_pl: gt_cp_corr,
				non_nan_mask_pl: non_nan_mask
			}

			pred_cp_corr, pred_cp_corr_top_k_idx, loss_listnet_val, loss_ce_val, loss_val, _ = sess.run([
				pred_cp_corr_tf, pred_cp_corr_top_k_idx_tf, loss_listnet_tf, loss_ce_tf, loss_tf, train_op
			], feed_dict=feed_dict)

			gt_cp_corr_num = np.sum(gt_cp_corr[0])
			gt_cp_corr_ratio = 1. * gt_cp_corr_num / gt_cp_corr.shape[-1]
			pred_cp_corr_acc = 1. * np.sum(gt_cp_corr[0][pred_cp_corr_top_k_idx[0]]) / len(pred_cp_corr_top_k_idx[0])
			
			info_dict = {
				'gt_cp_corr_num': gt_cp_corr_num,
				'gt_cp_corr_ratio': gt_cp_corr_ratio,
				'pred_cp_corr_acc': pred_cp_corr_acc,
			}
			loss_dict = {
				'loss_listnet_val': np.mean(loss_listnet_val[non_nan_idx]),
				'loss_ce_val': np.mean(loss_ce_val[non_nan_idx]),
				'loss_val': loss_val,
			}

			loss_dict_s2a = {
				'loss_o': np.mean(loss_o_val[non_nan_idx]),
				'loss_o_dist': np.mean(np.sqrt(2 * loss_o_val[non_nan_idx])),
				'loss_h': np.mean(loss_h_val[non_nan_idx]),
				'loss_h_dist': np.mean(np.sqrt(2 * loss_h_val[non_nan_idx])),
				'loss': np.mean((loss_o_val + loss_h_val)[non_nan_idx])
			}
			loss_dict_s1 = {
				'loss_transl': np.mean(loss_transl_val),
				'loss_transl_sqrt': np.sqrt(np.mean(loss_transl_val)),
				'loss_aa': np.mean(loss_aa_val),
				'loss_aa_sqrt': np.sqrt(np.mean(loss_aa_val)),
				'angle_diff': angle_diff,
				'angle_diff_min_pose_s2': angle_diff_min_pose_s2,
			}
			loss_tracker.add_dict(loss_dict)

			if log_it:
				write_tb(loss_dict, writer, 'train', total_ct)
				write_tb(loss_dict_s1, writer, 'train_s1', total_ct)
				write_tb(loss_dict_s2a, writer, 'train_s2a', total_ct)
				write_tb(info_dict, writer, 'train', total_ct)
				write_tb(cp_info_dict, writer, 'train_cp_data', total_ct)
				
			print('epoch {} iter {}/{} {}'.format(epoch_i, i, epoch_iter, loss_dict_to_str(loss_dict)))
			
			if (total_ct % args.model_save_freq == 0) and not args.no_save:
				save_model_generic(epoch_init + total_ct, model_folder, saver, sess)
				
		
		loss_dict_epoch = loss_tracker.stat()
		if can_write:
			write_tb(loss_dict_epoch, writer, 'train_epoch', total_ct)

		if (not args.no_eval) and (((epoch_i + 1) % args.eval_epoch_freq == 0) or args.run_test):
			eval_folder_dir = os.path.join(result_folder, 'eval')
			mkdir_if_not(eval_folder_dir)
			loss_tracker_test.reset()

			for i, batch_dict in enumerate(test_loader):
				pc_o = batch_dict['input1']
				pc_h = batch_dict['input2']
				gt_pose = batch_dict['output4']
				n_pose = batch_dict['n_pose']
				b_size = gt_pose.shape[0]

				pc_o_idx = batch_dict['pc_o_idx']
				pc_h_idx = batch_dict['pc_h_idx']
				cp_map_o_dir = batch_dict['cp_map_o_dir']
				cp_map_h_dir = batch_dict['cp_map_h_dir']
				cp_map_per_o_dir = batch_dict['cp_map_per_o_dir']
				cp_map_per_h_dir = batch_dict['cp_map_per_h_dir']

				pose_mult = np.ones((b_size, gt_pose.shape[1]))
				for ii in range(b_size):
					pose_mult[ii][n_pose[ii]:] = np.inf
				feed_dict_s1 = {
					pc_o_pl: pc_o[:, :, :3],
					pc_h_pl: pc_h[:, :, :3],
					gt_transl_pl: gt_pose[:, :, :3],
					gt_aa_pl: gt_pose[:, :, 3:],
					pose_mult_pl: pose_mult,
				}

				if i == 0:
					out_dir = os.path.join(eval_folder_dir, '{}_eval_epoch_{}_ct_{}.json'.format(file_name, str(epoch_i + 1), total_ct))
					eval_result_dict = {}
				for ii in range(args.eval_sample_n):
					z = np.random.normal(size=(pc_o.shape[0],1,args.z_dim))
					feed_dict_s1[z_pl] = z
					pred_transl, pred_aa, loss_transl_val, loss_aa_val, min_pose_idx = sess.run([
						pred_transl_tf, pred_aa_tf, loss_transl_tf, loss_aa_tf, min_pose_idx_tf], feed_dict=feed_dict_s1)
					gt_min_pose = np.squeeze(np.take_along_axis(gt_pose, np.expand_dims(min_pose_idx[:, 1:], axis=1), axis=1), axis=1)

					min_pose_idx_s2 = np.ones((b_size, 2), dtype=np.int32) * -1

					min_pose_idx_s2 = min_pose_idx
					gt_min_pose_s2 = np.squeeze(np.take_along_axis(gt_pose, np.expand_dims(min_pose_idx_s2[:, 1:], axis=1), axis=1), axis=1)

					# angle_diff = np.mean(angle_diff_batch(gt_min_pose[:, 3:], pred_aa, aa=True, degree=True))

					pc_combined = create_pc_combined_batch(pc_o, pc_h, pred_transl, pred_aa, aa=True)
					gt_cp_score_o, gt_cp_score_h, non_nan_idx = create_gt_cp_map(pc_o_idx, pc_h_idx, cp_map_o_dir, cp_map_h_dir, min_pose_idx_s2)
					gt_cp_map_per_o, gt_cp_map_per_h, non_nan_idx_per = load_gt_cp_per(pc_o_idx, pc_h_idx, cp_map_per_o_dir, cp_map_per_h_dir, min_pose_idx_s2)

					non_nan_idx = np.intersect1d(non_nan_idx, non_nan_idx_per)

					non_nan_mask = np.zeros((b_size), dtype=np.bool)
					non_nan_mask[non_nan_idx] = True
					feed_dict_s2a = {
						pc_combined_pl: pc_combined,
						gt_cp_score_o_pl: gt_cp_score_o,
						gt_cp_score_h_pl: gt_cp_score_h,
						non_nan_mask_pl: non_nan_mask
					}
					if len(non_nan_idx) == 0:
						continue

					pred_cp_score_o, pred_cp_score_h, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h, loss_o_val, loss_h_val = sess.run([
						pred_cp_score_o_tf, pred_cp_score_h_tf, pred_cp_top_k_idx_o_tf, pred_cp_top_k_idx_h_tf, loss_s2a_o_tf, loss_s2a_h_tf
					], feed_dict=feed_dict_s2a)

					gt_cp_corr, cp_info_dict = create_gt_cp_corr_preload_discretize(gt_cp_map_per_o, gt_cp_map_per_h, pred_cp_top_k_idx_o, pred_cp_top_k_idx_h, n_gt_sample=args.n_gt_sample)
					gt_cp_corr = gt_cp_corr.reshape((b_size, -1))
					feed_dict = {
						pc_combined_pl: pc_combined,
						gt_cp_corr_pl: gt_cp_corr,
						non_nan_mask_pl: non_nan_mask
					}

					pred_cp_corr, pred_cp_corr_top_k_idx, loss_listnet_val, loss_ce_val, loss_val = sess.run([
						pred_cp_corr_tf, pred_cp_corr_top_k_idx_tf, loss_listnet_tf, loss_ce_tf, loss_tf
					], feed_dict=feed_dict)

					gt_cp_corr_num = np.sum(gt_cp_corr[0])
					gt_cp_corr_ratio = 1. * gt_cp_corr_num / gt_cp_corr.shape[-1]
					pred_cp_corr_acc = 1. * np.sum(gt_cp_corr[0][pred_cp_corr_top_k_idx[0]]) / len(pred_cp_corr_top_k_idx[0])

					loss_dict = {
						'loss_o': np.mean(loss_o_val[non_nan_idx]),
						'loss_o_dist': np.mean(np.sqrt(2 * loss_o_val[non_nan_idx])),
						'loss_h': np.mean(loss_h_val[non_nan_idx]),
						'loss_h_dist': np.mean(np.sqrt(2 * loss_h_val[non_nan_idx])),
						
						'loss_listnet_val': np.mean(loss_listnet_val[non_nan_idx]),
						'loss_ce_val': np.mean(loss_ce_val[non_nan_idx]),
						'loss_val': loss_val,
						'pred_cp_corr_acc': pred_cp_corr_acc,
					}
					loss_tracker_test.add_dict(loss_dict)
					loss_tracker_test.add_dict(cp_info_dict)
					if i == 0:
						for jj in range(b_size):
							one_result_file_name = batch_dict['result_file_name'][jj]
							out_dict = {
								'z': z[jj].tolist(),
								'pc_o': pc_o[jj].tolist(),
								'pc_h': pc_h[jj].tolist(),
								'pc_combined': pc_combined[jj].tolist(),
								'pred_transl': pred_transl[jj].tolist(),
								'pred_aa': pred_aa[jj].tolist(),
								'gt_pose': gt_min_pose[jj].tolist(),
								'gt_pose_s2': gt_min_pose_s2[jj].tolist(),
								'min_pose_idx': int(min_pose_idx[jj][1]),
								'min_pose_idx_s2': int(min_pose_idx_s2[jj][1]),
								'gt_cp_score_o': gt_cp_score_o[jj].tolist(),
								'gt_cp_score_h': gt_cp_score_h[jj].tolist(),
								'pred_cp_score_o': pred_cp_score_o[jj].tolist(),
								'pred_cp_score_h': pred_cp_score_h[jj].tolist(),
								'loss_o': float(loss_o_val[jj]),
								'loss_h': float(loss_h_val[jj]),

								'gt_cp_map_per_o': gt_cp_map_per_o[jj].tolist(),
								'gt_cp_map_per_h': gt_cp_map_per_h[jj].tolist(),
								
								'pred_cp_top_k_idx_o': pred_cp_top_k_idx_o[jj].tolist(),
								'pred_cp_top_k_idx_h': pred_cp_top_k_idx_h[jj].tolist(),
								'gt_cp_corr': gt_cp_corr[jj].tolist(),
								'pred_cp_corr': pred_cp_corr[jj].tolist(), 
								'pred_cp_corr_top_k_idx': pred_cp_corr_top_k_idx[jj].tolist(),
								'loss_ce': float(loss_ce_val[jj]),
								'loss_listnet': float(loss_listnet_val[jj])
							}

							if not (one_result_file_name in eval_result_dict):
								eval_result_dict[one_result_file_name] = [out_dict]
							else:
								eval_result_dict[one_result_file_name].append(out_dict)

				if i == 0:
						save_json(out_dir, eval_result_dict)

				print('epoch {} iter {}/{} {}'.format(epoch_i, i, epoch_iter, loss_dict_to_str(loss_dict)))
			loss_dict_test_epoch = loss_tracker_test.stat()
	
			print('epoch {} test {}'.format(epoch_i, loss_dict_to_str(loss_dict_test_epoch)))

			if can_write:
				write_tb(loss_dict_test_epoch, writer, 'test_epoch', total_ct)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")

	parser.add_argument('--model_name', default='s2b_model')
	parser.add_argument('--comment', default='')
	parser.add_argument('--exp_name', default='exp_s2b')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--log_freq', type=int, default=2)


	parser.add_argument('--train_list', default='train_list')
	parser.add_argument('--test_list', default='test_list')
	parser.add_argument('--restrict_object_cat', default='')

	parser.add_argument('--run_test', action='store_true')
	parser.add_argument('--no_save', action='store_true')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--restore_model_name', default='s2b_model')
	parser.add_argument('--restore_model_epoch', type=int, default=57000)
	parser.add_argument('--max_epochs', type=int, default=10000)
	parser.add_argument('--eval_epoch_freq', type=int, default=1)
	parser.add_argument('--eval_sample_n', type=int, default=1)
	parser.add_argument('--model_save_freq', type=int, default=1000)
	parser.add_argument('--no_eval', action='store_true')

	parser.add_argument('--loss_transl_const', default=1000)

	parser.add_argument('--data_more_pose', action='store_true')

	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--z_dim', type=int, default=32)
	parser.add_argument('--top_k_o', type=int, default=128)
	parser.add_argument('--top_k_h', type=int, default=128)
	parser.add_argument('--n_gt_sample', type=int, default=128)

	parser.add_argument('--learning_rate', type=float, default=1e-4)

	parser.add_argument('--pretrain_s2a', action='store_true')
	parser.add_argument('--pretrain_s2a_folder', default='exp_s2a')
	parser.add_argument('--pretrain_s2a_model_name', default='s2a_model')
	parser.add_argument('--pretrain_s2a_epoch', default=60000, type=int)

	args = parser.parse_args()

	args.home_dir_data = os.path.abspath(args.home_dir_data)
	args.data_more_pose = True

	file_name = "{}".format(args.model_name)
	file_name += '_{}'.format(args.restrict_object_cat) if args.restrict_object_cat != '' else ''
	file_name += "_{}".format(args.comment) if args.comment != "" else ""
	file_name += '_overfit' if args.overfit else ''
	folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name

	if args.run_test:
		folder_name += '_test'
		
	result_folder = 'runs/{}'.format(folder_name) 
	if args.exp_name is not "":
		result_folder = 'runs/{}/{}'.format(args.exp_name, folder_name)
	if args.debug: 
		result_folder = 'runs/debug/{}'.format(folder_name)

	model_folder = os.path.join(result_folder, 'models')

	if not os.path.exists(model_folder):
		os.makedirs(result_folder)

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
		train_set = MyDataset(args.home_dir_data, train_list_dir, is_train=True, use_partial_pc=True, cp_s2a_soft=True, cp_s2b_soft=True, args=args)
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
							num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict, drop_last=True)
	
	test_set = MyDataset(args.home_dir_data, test_list_dir, is_train=False, use_partial_pc=True, cp_s2a_soft=True, cp_s2b_soft=True, args=args)

	test_loader = DataLoader(train_set if args.overfit else test_set, batch_size=args.batch_size, shuffle=True,
									num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict, drop_last=True)
	if not args.run_test:
		print('len of train {} len of test {}'.format(len(train_set), len(test_set)))
	else:
		print('len of train {} len of test {}'.format(len(test_set), len(test_set)))

	if not args.run_test:
		train(args, train_loader, test_loader, writer, result_folder, file_name)
	else:
		train(args, test_loader, test_loader, writer, result_folder, file_name)





