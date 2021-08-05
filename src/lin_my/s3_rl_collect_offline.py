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
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

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


from s3_rl_collect import calc_pose_cem_init, cem_transform_pc_batch, cem_eval, bullet_check

def train(args, train_set, train_loader, test_set, test_loader, writer, result_folder, file_name, extra_dict=None):
	model_folder = os.path.join(result_folder, 'models')
	can_write = not (writer is None)

	#stage 3
	pc_combined_pl, gt_succ_label_pl = s3_model.placeholder_inputs(args.batch_size, 4096, with_normal=False, args=args)
	pred_succ_cla_score_tf, end_points = s3_model.get_model(pc_combined_pl)

	pred_succ_cla_tf = tf.math.argmax(pred_succ_cla_score_tf, axis=-1)

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
	
	if args.pretrain_s3:
		pretrain_s3_folder = os.path.abspath(os.path.join(result_folder, '..', '..', args.pretrain_s3_folder, args.pretrain_s3_model_name, 'models'))
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
        max_action=args.cem_max_transl,
        max_action_aa=args.cem_max_aa,
        sigma_init=args.cem_sigma_init_transl,
        sigma_init_aa=args.cem_sigma_init_aa,
        pop_size=args.cem_pop_size,
        damp=args.cem_damp_transl,
        damp_limit=args.cem_damp_limit_transl,
        damp_aa=args.cem_damp_aa,
        damp_limit_aa=args.cem_damp_limit_aa,
        parents=args.cem_parents,		
	)
	
	fcl_hook_dict = extra_dict['fcl_hook_dict']
	fcl_object_dict = extra_dict['fcl_object_dict']

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

	preload_dict_all = load_json(args.preload_pose_dir)
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
		if not run_test:
			loader = train_loader
			dataset = train_set
		else:
			loader = test_loader
			dataset = test_set
		
		info_dict_all = {}

		for i, batch_dict in enumerate(loader):
			total_ct += 1

			if (i + 1) % 20 == 0:
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
			
			cem_init_transl = np.zeros((b_size, 3))
			cem_init_aa = np.zeros((b_size, 3))
			cem_rotation_center_o = np.zeros((b_size, 3))

			for ii in range(b_size):
				cem_init_transl[ii] = preload_dict_all[result_file_name[ii]]['cem_init_transl']
				cem_init_aa[ii] = preload_dict_all[result_file_name[ii]]['cem_init_aa']
				cem_rotation_center_o[ii] = preload_dict_all[result_file_name[ii]]['cem_rotation_center_o']

			# stage 3

			pc_o_cem_init = transform_pc_batch(pc_o[:, :, :3], cem_init_transl, cem_init_aa)
			cem_eval_partial = partial(cem_eval,
				pc_o=pc_o_cem_init,
				pc_h=pc_h[:, :, :3],
				rotation_center_o=cem_rotation_center_o,
				sess_tf=sess,
				pc_combined_pl=pc_combined_pl,
				pred_succ_cla_score_tf=pred_succ_cla_score_tf,
			)
			_, cem_out_pose, cem_search_info_dict = cem.search(
				b_size,
				np.array([[0, 0, 0, 1e-6, 0, 0]] * b_size), 
				cem_eval_partial,
				n_iter=args.cem_n_iter,
				elitism=True,
				visualize=False,
				visualize_func=None,
			)
			cem_out_transl = cem_out_pose[:, :3]
			cem_out_aa = cem_out_pose[:, 3:]

			# convert the cem output pose to object pose
			pc_o_cem = cem_transform_pc_batch(pc_o_cem_init[:, :, :3], cem_rotation_center_o, cem_out_transl, cem_out_aa)
			
			final_pred_transl, final_pred_aa = best_fit_transform_batch(pc_o[:, :, :3], pc_o_cem)

			# bullet check
			bullet_check_func = partial(
				bullet_check,
				transl=final_pred_transl,
				aa=final_pred_aa,
				p_list=p_list,
				result_file_name=result_file_name,
				hook_urdf=hook_urdf,
				object_urdf=object_urdf,
				fcl_hook_model=fcl_hook_model,
				fcl_object_model=fcl_object_model,
			)

			bullet_succ = np.zeros((b_size))
			for bi, (flag_tmp, bullet_final_transl, bullet_final_quat) in enumerate(pool.imap(bullet_check_func, range(b_size))):
				succ = 1. if flag_tmp else 0.
				bullet_succ[bi] = succ

				hook_name, object_name = split_result_file_name(result_file_name[bi])
				if not run_test:
					replay_buffer.add(object_name, hook_name, np.append(final_pred_transl[bi], final_pred_aa[bi]), succ)

			info_dict = {}
			for ii in range(b_size):
				info_dict[result_file_name[ii]] = preload_dict_all[result_file_name[ii]]
				info_dict[result_file_name[ii]].update({
					'succ': bullet_succ[ii],
					'final_pred_transl': final_pred_transl[ii].tolist(),
					'final_pred_aa': final_pred_aa[ii].tolist(),
					'cem_out_transl': cem_out_pose[ii, :3].tolist(),
					'cem_out_aa': cem_out_pose[ii, 3:].tolist(),
					# 'cem_elite_pose': cem_elite_pose[ii].tolist()
				})
				for tmp_key in cem_search_info_dict: 
					info_dict[result_file_name[ii]][tmp_key] = cem_search_info_dict[tmp_key][ii].tolist()

			info_dict_all.update(info_dict)
			
			loss_dict = {
				'bullet_succ_acc': np.mean(bullet_succ)
			}
			loss_tracker.add_dict(loss_dict)

			if log_it:
				write_tb(loss_dict, writer, 'test' if run_test else 'train', total_ct)
			print('epoch {} iter {}/{} {}'.format(epoch_i, i, epoch_iter, loss_dict_to_str(loss_dict)))

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

			if (total_ct % args.model_save_freq == 0) and not args.no_save:
				save_model_generic(epoch_init + total_ct, model_folder, saver, sess)

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
	parser.add_argument("--home_dir_data", default="'/scr1/new_hang'")
	parser.add_argument('--pointset_dir', default='/scr2/')
	parser.add_argument('--bohg4', action='store_true')
	parser.add_argument('--no_vis', action='store_true')

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


	parser.add_argument('--pretrain_s3', action='store_true')
	parser.add_argument('--pretrain_s3_folder', default='exp_s3')
	parser.add_argument('--pretrain_s3_model_name', default='Feb19_14-26-47_s3_classifier_model_new_data')
	parser.add_argument('--pretrain_s3_epoch', default=750000, type=int)

	parser.add_argument('--preload_pose_dir', default='/scr1/yifan/geo-hook/lin_my/runs/exp_s3/Mar13_23-04-53_s3_rl_collect_20_rand/eval/s3_rl_collect_20_rand_eval_epoch_1_train.json')
	args = parser.parse_args()

	args.data_more_pose = True

	if args.bohg4:
		args.pointset_dir = '/scr1/yifan'
		args.no_vis = True
		args.home_dir_data = '/scr1/yifan/hang'

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

	if args.debug:
		args.s3_train_name = 'debug'
	else:
		args.s3_train_name += "_{}".format(args.comment) if args.comment != "" else ""
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
		train_set = MyDataset(args.home_dir_data, train_list_dir, is_train=True, use_partial_pc=True, use_fcl=(not args.no_fcl), args=args)
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
							num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict, drop_last=True)
	
	if not args.no_eval:
		test_set = MyDataset(args.home_dir_data, test_list_dir, is_train=False, use_partial_pc=True, use_fcl=(not args.no_fcl), args=args)

		test_loader = DataLoader(train_set if args.overfit else test_set, batch_size=args.batch_size, shuffle=True,
									num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict, drop_last=True)
	else:
		test_set = None
		test_loader = None
	# if not args.run_test:
	# 	print('len of train {} len of test {}'.format(len(train_set), len(test_set)))
	# else:
	# 	print('len of train {} len of test {}'.format(len(test_set), len(test_set)))

	extra_dict = {
		'fcl_object_dict': train_set.fcl_object_dict,
		'fcl_hook_dict': train_set.fcl_hook_dict,
	}
	if not args.run_test:
		train(args, train_set, train_loader, test_set, test_loader, writer, result_folder, file_name, extra_dict=extra_dict)
	else:
		train(args, test_set, test_loader, test_set, test_loader, writer, result_folder, file_name, extra_dict=extra_dict)





