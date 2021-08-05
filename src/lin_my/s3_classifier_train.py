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
		
	total_ct = 0

	# p_env = p_Env(args.home_dir_data, gui=False, physics=False)
	
	for epoch_i in range(args.max_epochs):
		loss_tracker.reset()
		for i, batch_dict in enumerate(train_loader):
			
			if not args.run_test:
				total_ct += 1
				log_it = ((total_ct % args.log_freq ) == 0) and can_write

				data_pc_combined = batch_dict['input3']
				gt_succ_label = batch_dict['label']
				pc_combined = data_pc_combined[:, :, [0, 1, 2, -1]]
				gt_normal = data_pc_combined[:, :, 3:6]

				feed_dict = {
					pc_combined_pl: pc_combined,
					gt_succ_label_pl: gt_succ_label
				}

				pred_succ_cla, pred_succ_cla_score, loss_succ_cla_val, _ = sess.run([
					pred_succ_cla_tf, pred_succ_cla_score_tf, loss_succ_cla_tf, train_op 
				], feed_dict=feed_dict)
					
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
				
			
			if (i == len(train_loader) - 1) or (i > 0 and i % 3000 == 0) or args.run_test:
				if (not args.no_eval) and (((epoch_i + 1) % args.eval_epoch_freq == 0) or args.run_test):
					eval_folder_dir = os.path.join(result_folder, 'eval')
					mkdir_if_not(eval_folder_dir)
					
					total_ct_test = 0
					loss_tracker_test.reset()
					for i, batch_dict in enumerate(test_loader):
						data_pc_combined = batch_dict['input3']
						gt_succ_label = batch_dict['label']
						pc_combined = data_pc_combined[:, :, [0, 1, 2, -1]]
						gt_normal = data_pc_combined[:, :, 3:6]
						b_size = pc_combined.shape[0]
						feed_dict = {
							pc_combined_pl: pc_combined,
							gt_succ_label_pl: gt_succ_label
						}
						pred_succ_cla, pred_succ_cla_score, loss_succ_cla_val = sess.run([
							pred_succ_cla_tf, pred_succ_cla_score_tf, loss_succ_cla_tf 
						], feed_dict=feed_dict)

						acc_dict = get_acc(gt_succ_label, pred_succ_cla)

						loss_dict = {
							'loss_succ_cla': loss_succ_cla_val,
							'acc_succ_cla': acc_dict['acc'],
							'acc_succ_cla_pos': acc_dict['acc_cla_pos'],
							'acc_succ_cla_neg': acc_dict['acc_cla_neg'],
							'acc_succ_precision': acc_dict['precision'],
							'acc_succ_recall': acc_dict['recall'],
						}
						loss_tracker_test.add_dict(loss_dict)
						print('epoch {} iter {}/{} {}'.format(epoch_i, i, epoch_iter, loss_dict_to_str(loss_dict)))

						if i == 0:
							out_dir = os.path.join(eval_folder_dir, '{}_eval_epoch_{}_ct_{}.json'.format(file_name, str(epoch_i + 1), total_ct))
							eval_result_dict = {}
							for jj in range(b_size):
								one_result_file_name = batch_dict['result_file_name'][jj]
								out_dict = {
									'pc_combined': pc_combined[jj].tolist(),
									'gt_normal': gt_normal[jj].tolist(),
									'pred_succ_cla': int(pred_succ_cla[jj]),
									'gt_succ_label': int(gt_succ_label[jj]),
									'pred_succ_cla_score': (pred_succ_cla_score[jj]).tolist(),
								}
								eval_result_dict[one_result_file_name] = out_dict
							save_json(out_dir, eval_result_dict)
							
						if i > 1000:
							break
					loss_dict_test_epoch = loss_tracker_test.stat()
			
					print('epoch {} test {}'.format(epoch_i, loss_dict_to_str(loss_dict_test_epoch)))

					if can_write:
						write_tb(loss_dict_test_epoch, writer, 'test_epoch', total_ct)
			if args.run_test:
				break

		loss_dict_epoch = loss_tracker.stat()
		if can_write:
			write_tb(loss_dict_epoch, writer, 'train_epoch', total_ct)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument('--pointset_dir', default='/scr2/')
	parser.add_argument('--bohg4', action='store_true')
	parser.add_argument('--no_vis', action='store_true')

	parser.add_argument('--model_name', default='s3_classifier_model')
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

	args = parser.parse_args()

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
	if not args.run_test:
		print('len of train {} len of test {}'.format(len(train_set), len(test_set)))
	else:
		print('len of train {} len of test {}'.format(len(test_set), len(test_set)))

	if not args.run_test:
		train(args, train_loader, test_loader, writer, result_folder, file_name)
	else:
		train(args, test_loader, test_loader, writer, result_folder, file_name)





