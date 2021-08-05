import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import random
from classifier_dataset_new import MyDataset 
import os
import time
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
import datetime
random.seed(2)
torch.manual_seed(2)
np.random.seed(4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from train_helper import *
from rotation_lib import *
from bullet_helper import *
import s3_classifier_model as my_model
import s3_replay_buffer_pose as ReplayBuffer
from s2_utils import *

	
def train(args, train_loader, test_loader, writer, result_folder, file_name):
	model_folder = os.path.join(result_folder, 'models')
	can_write = not (writer is None)

	pc_combined_pl, gt_succ_label_pl = my_model.placeholder_inputs(args.batch_size, 4096, with_normal=False, args=args)
	pred_succ_cla_score_tf, end_points = my_model.get_model(pc_combined_pl)

	pred_succ_cla_tf = tf.math.argmax(pred_succ_cla_score_tf, axis=-1)
	loss_succ_cla_tf = my_model.get_loss(pred_succ_cla_score_tf, gt_succ_label_pl, end_points)

	train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_succ_cla_tf)

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
	
	if args.restore_model_epoch != -1:
		epoch_init = args.restore_model_epoch
		restore_model_folder = os.path.abspath(os.path.join(result_folder, '..', args.restore_model_name, 'models'))
		restore_model_generic(epoch_init, restore_model_folder, saver, sess)
	if args.pretrain_s3:
		pretrain_s3_folder = os.path.abspath(os.path.join(result_folder, '..', '..', args.pretrain_s3_folder, args.pretrain_s3_model_name, 'models'))
		restore_model_generic(args.pretrain_s3_epoch, pretrain_s3_folder, saver, sess)
	else:
		restore_model_s3_second_last(args.s3_model_dir, sess)

		
	total_ct = 0

	# p_env = p_Env(args.home_dir_data, gui=False, physics=False)
	
	if not args.run_test:
		s3_buffer_dir_arr = [os.path.join(args.s3_buffer_dir, tmp) for tmp in os.listdir(args.s3_buffer_dir)]
		print(args.s3_buffer_dir)
		print(s3_buffer_dir_arr)
		saved_buffer_name_list = [get_2nd_last_dir(tmp) for tmp in s3_buffer_dir_arr]
		replay_buffer = ReplayBuffer.combine_replay_buffer(saved_buffer_name_list, train_set, args.home_dir_data, preload_data=False if args.no_preload else True)

	for epoch_i in range(args.max_epochs):
		loss_tracker.reset()
		for i in range(len(train_set) // args.batch_size):
			total_ct += 1
			log_it = ((total_ct % args.log_freq ) == 0) and can_write

			pc_combined, gt_succ_label = replay_buffer.sample(args.batch_size)
			feed_dict = {
				pc_combined_pl: pc_combined,
				gt_succ_label_pl: gt_succ_label
			}

			pred_succ_cla, pred_succ_cla_score, loss_succ_cla_val, _ = sess.run([
				pred_succ_cla_tf, pred_succ_cla_score_tf, loss_succ_cla_tf, train_op 
			], feed_dict=feed_dict)
			# from scipy.special import softmax
			# print('score', pred_succ_cla_score[0], softmax(pred_succ_cla_score[0:1], axis=-1))
			acc_dict = get_acc(gt_succ_label, pred_succ_cla)

			loss_dict = {
				'loss_succ_cla': loss_succ_cla_val,
				'acc_succ_cla': acc_dict['acc'],
				'acc_succ_cla_pos': acc_dict['acc_cla_pos'],
				'acc_succ_cla_neg': acc_dict['acc_cla_neg'],
				'acc_succ_precision': acc_dict['precision'],
				'acc_succ_recall': acc_dict['recall'],
			}
			loss_tracker.add_dict(loss_dict)

			if log_it:
				write_tb(loss_dict, writer, 'train', total_ct)

			print('epoch {} iter {}/{} {}'.format(epoch_i, i, epoch_iter, loss_dict_to_str(loss_dict)))
			
			if (total_ct % args.model_save_freq == 0) and not args.no_save:
				save_model_generic(epoch_init + total_ct, model_folder, saver, sess)

			# periodically load buffer
			if int(total_ct) % args.s3_buffer_freq == 0:
				saved_buffer_name_list = [get_2nd_last_dir(tmp) for tmp in s3_buffer_dir_arr]
				replay_buffer = ReplayBuffer.combine_replay_buffer(saved_buffer_name_list, train_set, args.home_dir_data, preload_data=False if args.no_preload else True)
				print('new replay buffer', replay_buffer.buffer_succ.size, replay_buffer.buffer_fail.size)
				
			
			# periodically save s3 model
			if int(total_ct) % args.s3_model_freq == 0:
				save_model_generic(epoch_init + total_ct, args.s3_model_dir, saver, sess)

			if (total_ct % args.model_save_freq == 0) and not args.no_save:
				save_model_generic(epoch_init + total_ct, model_folder, saver, sess)


		loss_dict_epoch = loss_tracker.stat()
		if can_write:
			write_tb(loss_dict_epoch, writer, 'train_epoch', total_ct)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")

	parser.add_argument('--model_name', default='s3_rl_train')
	parser.add_argument('--comment', default='')
	parser.add_argument('--exp_name', default='exp_s3')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--log_freq', type=int, default=10)

	parser.add_argument('--train_list', default='train_list')
	parser.add_argument('--test_list', default='test_list')
	parser.add_argument('--restrict_object_cat', default='')

	parser.add_argument('--run_test', action='store_true')
	parser.add_argument('--no_save', action='store_true')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--restore_model_name', default='')
	parser.add_argument('--restore_model_epoch', type=int, default=-1)
	parser.add_argument('--max_epochs', type=int, default=10000)
	parser.add_argument('--eval_epoch_freq', type=int, default=1)
	parser.add_argument('--model_save_freq', type=int, default=3000)
	parser.add_argument('--no_eval', action='store_true')

	parser.add_argument('--full_pc', action='store_true')

	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--learning_rate', type=float, default=1e-4)

	parser.add_argument('--s3_train_folder_dir', default='/juno/downloads/new_hang_training/')
	parser.add_argument('--s3_train_name', default='s3')
	parser.add_argument('--s3_buffer_dir', default='')
	parser.add_argument('--s3_model_dir', default='')
	parser.add_argument('--s3_buffer_freq', default=1000, type=int)
	parser.add_argument('--s3_model_freq', default=1000, type=int)

	parser.add_argument('--no_preload', action='store_true')

	parser.add_argument('--pretrain_s3', action='store_true')
	parser.add_argument('--pretrain_s3_folder', default='exp_s3')
	parser.add_argument('--pretrain_s3_model_name', default='Mar19_03-43-24_s3_rl_train_all')
	parser.add_argument('--pretrain_s3_epoch', default=18000, type=int)

	parser.add_argument('--lin', action='store_true')

	args = parser.parse_args()
	args.home_dir_data = os.path.abspath(args.home_dir_data)

	if args.run_test:
		assert False, 'not implemented yet'	

		
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
	args.s3_buffer_dir = os.path.join(s3_train_dir, 'buffers')
	args.s3_model_dir = os.path.join(s3_train_dir, 'models')
	

	print("---------------------------------------")
	print("Model Name: {}, Train List: {}, Test List: {}".format(args.model_name, args.train_list, args.test_list))
	print("---------------------------------------")

	if args.run_test:
		print("Restore Model: {}".format(args.restore_model_name))
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
	if not args.run_test:
		train_set = MyDataset(args.home_dir_data, train_list_dir, is_train=True, use_partial_pc=False if args.full_pc else True, args=args)
		train_set.build_sampler()
		train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_set.sampler,
							num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict)
	
	test_set = MyDataset(args.home_dir_data, test_list_dir, is_train=False, use_partial_pc=False if args.full_pc else True, args=args)

	test_loader = DataLoader(train_set if args.overfit else test_set, batch_size=args.batch_size, shuffle=True,
									num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict)
	# if not args.run_test:
	# 	print('len of train {} len of test {}'.format(len(train_set), len(test_set)))
	# else:
	# 	print('len of train {} len of test {}'.format(len(test_set), len(test_set)))

	if not args.run_test:
		train(args, train_loader, test_loader, writer, result_folder, file_name)
	else:
		train(args, test_loader, test_loader, writer, result_folder, file_name)





