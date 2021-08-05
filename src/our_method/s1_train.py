import tensorflow as tf
import numpy as np
import sys
import random
from hang_dataset import MyDataset 
import os
import time
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch
# from torch.utils.tensorboard import SummaryWriter
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
import s1_model as my_model

def train(args, train_loader, test_loader, writer, result_folder, file_name):
	model_folder = os.path.join(result_folder, 'models')
	can_write = not (writer is None)
	pc_o_pl, pc_h_pl, z_pl, gt_transl_pl, gt_aa_pl, pose_mult_pl, pose_ct_pl = my_model.placeholder_inputs(args.batch_size, 4096, args)

	pred_transl_tf, pred_aa_tf, end_points = my_model.get_model(pc_o_pl, pc_h_pl, z_pl)
	loss_transl_tf, loss_aa_tf, min_pose_idx_tf, loss_per_pose_tf = my_model.get_loss(pred_transl_tf, pred_aa_tf, gt_transl_pl, gt_aa_pl, pose_mult_pl, args.loss_transl_const, end_points)
	loss_mult_tf = my_model.get_loss_mult(pred_transl_tf, pred_aa_tf, gt_transl_pl, gt_aa_pl, pose_ct_pl, args.loss_transl_const, end_points)

	optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

	# gradient averaging
	# https://gchlebus.github.io/2018/06/05/gradient-averaging.html
	grads_and_vars = optimizer.compute_gradients(loss_mult_tf)
	avg_grads_and_vars = []
	grad_placeholders = []
	for grad, var in grads_and_vars:
		grad_ph = tf.placeholder(grad.dtype, grad.shape)
		grad_placeholders.append(grad_ph)
		avg_grads_and_vars.append((grad_ph, var))
	grad_op = [x[0] for x in grads_and_vars]

	train_op = optimizer.apply_gradients(avg_grads_and_vars)
	gradients = [] # list to store gradients

	init_op = tf.group(tf.global_variables_initializer(),
				tf.local_variables_initializer())

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.Session(config=config)
	sess.run(init_op)
	saver = tf.train.Saver(max_to_keep=1000)

	epoch_init = 0
	if not train_loader is None:
		epoch_iter = len(train_loader)
	if args.restore_model_epoch != -1:
		epoch_init = args.restore_model_epoch
		restore_model_folder = os.path.abspath(os.path.join(result_folder, '..', args.restore_model_name, 'models'))
		restore_model_generic(epoch_init, restore_model_folder, saver, sess)
		
	total_ct = 0

	p_env = p_Env(args.home_dir_data, gui=False, physics=False)
	
	from scipy.optimize import linear_sum_assignment
	
	for epoch_i in range(args.max_epochs):
		loss_sum = 0
		loss_transl_sum = 0
		loss_aa_sum = 0
		angle_diff_sum = 0

		for i, batch_dict in enumerate(train_loader):
			if args.run_test:
				break
			total_ct += 1
			log_it = ((total_ct % args.log_freq ) == 0) and can_write
			pc_o = batch_dict['input1']
			pc_h = batch_dict['input2']
			gt_pose = batch_dict['output4']
			n_pose = batch_dict['n_pose']
			pose_mult = np.ones((gt_pose.shape[0], gt_pose.shape[1]))
			for ii in range(gt_pose.shape[0]):
				pose_mult[ii][n_pose[ii]:] = np.inf
			feed_dict = {
				pc_o_pl: pc_o[:, :, :3],
				pc_h_pl: pc_h[:, :, :3],
				gt_transl_pl: gt_pose[:, :, :3],
				gt_aa_pl: gt_pose[:, :, 3:],
				pose_mult_pl: pose_mult,
			}

			# meshgrid z test
			# if total_ct % 20 == 0:
			# if True:
			if 0:
				z_plots_folder = os.path.join(result_folder, 'plot_z')
				mkdir_if_not(z_plots_folder)

				z_one = np.arange(-3, 3, 0.2)
				z_xx, z_yy = np.meshgrid(z_one, z_one)
				z_grid = np.stack([z_xx.flatten(), z_yy.flatten()], axis=1)
				z_grid = np.expand_dims(z_grid, axis=1)
				min_pose_idx_grid = np.zeros((z_grid.shape[0]))
				loss_grid = np.zeros((z_grid.shape[0]))
				pred_grid = np.zeros((z_grid.shape[0], 6))

				for k in range(0, z_grid.shape[0], args.batch_size):
					z_tmp = z_grid[k:k+args.batch_size]
					n_rep = z_tmp.shape[0]
					feed_dict_tmp = {pl: np.repeat(tmp[0:1], n_rep, axis=0) for pl, tmp in feed_dict.items()}
					feed_dict_tmp[z_pl] = z_tmp
					
					pred_transl_tmp, pred_aa_tmp, loss_transl_val_tmp, loss_aa_val_tmp, min_pose_idx_tmp = sess.run([
						pred_transl_tf, pred_aa_tf, loss_transl_tf, loss_aa_tf, min_pose_idx_tf,
					], feed_dict=feed_dict_tmp)

					min_pose_idx_grid[k:k+args.batch_size] = min_pose_idx_tmp[:, 1]
					loss_grid[k:k+args.batch_size] = loss_transl_val_tmp * args.loss_transl_const + loss_aa_val_tmp
					pred_grid[k:k+args.batch_size, :3] = pred_transl_tmp
					pred_grid[k:k+args.batch_size, 3:] = pred_aa_tmp
				
				min_pose_idx_grid = min_pose_idx_grid.reshape((z_one.shape[0], z_one.shape[0]))
				loss_grid = loss_grid.reshape((z_one.shape[0], z_one.shape[0]))

				z_plots_dir_half = os.path.join(z_plots_folder, '{}_{}'.format(total_ct + epoch_init, batch_dict['result_file_name'][0]))
				pred_grid = pred_grid.reshape((z_one.shape[0], z_one.shape[0], 6))
				plt.clf()
				plt.pcolormesh(min_pose_idx_grid, vmin=0, vmax=gt_pose.shape[1])
				plt.colorbar()
				plt.savefig(z_plots_dir_half + '.jpg')
				plt.clf()
				plt.pcolormesh(loss_grid, vmin=0, vmax=np.sort(loss_grid, axis=None)[-4])
				plt.colorbar()
				plt.savefig(z_plots_dir_half + '_loss.jpg')
				
				plot_every_n = 5
				fig, axs = plt.subplots(len(range(0, pred_grid.shape[0], plot_every_n)), len(range(0, pred_grid.shape[0], plot_every_n)))
				for ii in range(0, pred_grid.shape[0], plot_every_n):
					for jj in range(0, pred_grid.shape[1], plot_every_n):
						pred_pose_tmp = pred_grid[ii][jj]
						p_env.load_pair_w_pose(batch_dict['result_file_name'][0], pred_pose_tmp[:3], pred_pose_tmp[3:], aa=True)
						img_tmp = p_env.photo()

						ax_i = int(ii/plot_every_n)
						ax_j = int(jj/plot_every_n)
						axs[ax_i, ax_j].imshow(img_tmp)
						axs[ax_i, ax_j].axis('off')

				plt_whitespace(plt)
				plt.savefig(z_plots_dir_half + '_pose.jpg',
					dpi=1000, bbox_inches = 'tight', pad_inches = 0)

				plt.clf()

				gt_photo_dir = os.path.join(z_plots_folder, 'gt_pose.jpg')
				if not os.path.exists(gt_photo_dir):
					n_in_row = 6
					n_row = int(math.ceil(n_pose[0] * 1. / n_in_row))
					fig, axs = plt.subplots(n_row, n_in_row)
					for ii in range(n_pose[0]):
						p_env.load_pair_w_pose(batch_dict['result_file_name'][0], gt_pose[0][ii][:3], gt_pose[0][ii][3:], aa=True)
						img_tmp = p_env.photo()
						ax_i = ii // n_in_row
						ax_j = ii % n_in_row
						if n_row > 1:
							axs[ax_i, ax_j].imshow(img_tmp)
							axs[ax_i, ax_j].axis('off')
						else:
							axs[ax_j].imshow(img_tmp)
							axs[ax_j].axis('off')

					plt_whitespace(plt)
					plt.savefig(gt_photo_dir,
						dpi=1000, bbox_inches = 'tight', pad_inches = 0)

			n_z_sample = args.n_z_sample
			pred_transl_all = np.zeros((pc_o.shape[0], n_z_sample, 3))
			pred_aa_all = np.zeros((pc_o.shape[0], n_z_sample, 3))
			z_all = np.zeros((pc_o.shape[0], n_z_sample, 1, args.z_dim))
			loss_per_pose_all = np.zeros((pc_o.shape[0], n_z_sample, gt_pose.shape[1]))

			for ii in range(n_z_sample):
				z_tmp = np.random.normal(size=(pc_o.shape[0],1,args.z_dim))
				feed_dict[z_pl] = z_tmp

				if ii == 0:
					pred_transl, pred_aa, loss_transl_val, loss_aa_val, min_pose_idx, loss_per_pose = sess.run([
						pred_transl_tf, pred_aa_tf, loss_transl_tf, loss_aa_tf, min_pose_idx_tf, loss_per_pose_tf
					], feed_dict=feed_dict)
					gt_min_pose = np.squeeze(np.take_along_axis(gt_pose, np.expand_dims(min_pose_idx[:, 1:], axis=1), axis=1), axis=1)
					angle_diff = np.mean(angle_diff_batch(gt_min_pose[:, 3:], pred_aa, aa=True, degree=True))
			
				else:
					pred_transl, pred_aa, loss_per_pose = sess.run([
						pred_transl_tf, pred_aa_tf, loss_per_pose_tf
					], feed_dict=feed_dict)

				pred_transl_all[:, ii, :] = pred_transl
				pred_aa_all[:, ii, :] = pred_aa
				z_all[:, ii, :, :] = z_tmp
				loss_per_pose_all[:, ii, :] = loss_per_pose

			loss_mult_all = np.zeros((pc_o.shape[0]))
			for ii in range(pc_o.shape[0]):
				n_pose_tmp = n_pose[ii]
				loss_per_pose_tmp = loss_per_pose_all[ii, :n_z_sample, :n_pose_tmp]
				
				pose_ct = np.zeros((n_z_sample, n_pose_tmp))
				min_idx_0 = np.expand_dims(np.argmin(loss_per_pose_tmp, axis=0), axis=0)
				min_idx_1 = np.expand_dims(np.argmin(loss_per_pose_tmp, axis=1), axis=1)
				pose_ct_tmp = np.take_along_axis(pose_ct, min_idx_0, axis=0)
				pose_ct_tmp += 1
				np.put_along_axis(pose_ct, min_idx_0, pose_ct_tmp, axis=0)

				pose_ct_tmp = np.take_along_axis(pose_ct, min_idx_1, axis=1)
				pose_ct_tmp += 1
				np.put_along_axis(pose_ct, min_idx_1, pose_ct_tmp, axis=1)

				pc_o_in = np.repeat(pc_o[ii:ii+1], n_z_sample, axis=0)
				pc_h_in = np.repeat(pc_h[ii:ii+1], n_z_sample, axis=0)
				gt_pose_in = np.repeat(gt_pose[ii:ii+1, :n_pose_tmp], n_z_sample, axis=0)
				z_in = z_all[ii, :n_z_sample]
				feed_dict_tmp = {
					pc_o_pl: pc_o_in[:, :, :3],
					pc_h_pl: pc_h_in[:, :, :3],
					gt_transl_pl: gt_pose_in[:, :, :3],
					gt_aa_pl: gt_pose_in[:, :, 3:],
					z_pl: z_in,
					pose_ct_pl: pose_ct
				}
				loss_mult_val, grads = sess.run([loss_mult_tf, grad_op], feed_dict=feed_dict_tmp)
				gradients.append(grads)

				loss_mult_all[ii] = loss_mult_val

			# aggregate gradients & train
			feed_dict_tmp = {}
			for ii, placeholder in enumerate(grad_placeholders):
				feed_dict_tmp[placeholder] = np.stack([g[ii] for g in gradients], axis=0).mean(axis=0)
			sess.run(train_op, feed_dict=feed_dict_tmp)
			gradients = []

			loss_transl_val = np.mean(loss_transl_val)
			loss_aa_val = np.mean(loss_aa_val)
			loss_sum += loss_transl_val + loss_aa_val

			loss_transl_sum += loss_transl_val
			loss_aa_sum += loss_aa_val
			angle_diff_sum += angle_diff

			if log_it:
				writer.add_scalar('train/loss_transl', loss_transl_val, total_ct)
				writer.add_scalar('train/loss_transl_sqrt', np.sqrt(loss_transl_val), total_ct)
				writer.add_scalar('train/loss_aa', loss_aa_val, total_ct)
				writer.add_scalar('train/loss_aa_sqrt', np.sqrt(loss_aa_val), total_ct)
				writer.add_scalar('train/loss', loss_transl_val + loss_aa_val, total_ct)
				writer.add_scalar('train/loss_mult', np.mean(loss_mult_all), total_ct)
				writer.add_scalar('train/angle_diff', angle_diff, total_ct)

			print('epoch {} iter {}/{} loss_transl {:.4f},{:.4f} loss_aa {:.4f},{:.4f}'.format(epoch_i, i, epoch_iter, loss_transl_val, np.sqrt(loss_transl_val), loss_aa_val, np.sqrt(loss_aa_val)))
			
			if (total_ct % args.model_save_freq == 0) and not args.no_save:
				save_model_generic(epoch_init + total_ct, model_folder, saver, sess)
				
		if can_write:
			writer.add_scalar('train_epoch/loss', loss_sum / epoch_iter, total_ct)
			writer.add_scalar('train_epoch/loss_transl', loss_transl_sum / epoch_iter, total_ct)
			writer.add_scalar('train_epoch/loss_transl_sqrt', np.sqrt(loss_transl_sum / epoch_iter), total_ct)
			writer.add_scalar('train_epoch/loss_aa', loss_aa_sum / epoch_iter, total_ct)
			writer.add_scalar('train_epoch/loss_aa_sqrt', np.sqrt(loss_aa_sum / epoch_iter), total_ct)
			writer.add_scalar('train_epoch/angle_diff', angle_diff_sum / epoch_iter, total_ct)
		
		if (not args.no_eval) and (((epoch_i + 1) % args.eval_epoch_freq == 0) or args.run_test):
			eval_result_dict = {}
			eval_result_dict['params'] = {
				'total_ct': total_ct
			}
			eval_file_name = '{}_eval_{}.json'.format(file_name, str(epoch_i+1))
			eval_file_dir = os.path.join(result_folder, eval_file_name)

			loss_sum_test = 0
			loss_transl_sum_test = 0
			loss_aa_sum_test = 0
			angle_diff_sum_test = 0
			total_ct_test = 0
			for i, batch_dict in enumerate(test_loader):
				pc_o = batch_dict['input1']
				pc_h = batch_dict['input2']
				gt_pose = batch_dict['output4']
				n_pose = batch_dict['n_pose']


				pose_mult = np.ones((gt_pose.shape[0], gt_pose.shape[1]))
				for ii in range(gt_pose.shape[0]):
					pose_mult[ii][n_pose[ii]:] = np.inf
				feed_dict = {
					pc_o_pl: pc_o[:, :, :3],
					pc_h_pl: pc_h[:, :, :3],
					gt_transl_pl: gt_pose[:, :, :3],
					gt_aa_pl: gt_pose[:, :, 3:],
					pose_mult_pl: pose_mult,
				}

				for ii in range(args.eval_sample_n):
					z = np.random.normal(size=(pc_o.shape[0],1,args.z_dim))
					feed_dict[z_pl] = z
					pred_transl, pred_aa, loss_transl_val, loss_aa_val, min_pose_idx = sess.run([
						pred_transl_tf, pred_aa_tf, loss_transl_tf, loss_aa_tf, min_pose_idx_tf], feed_dict=feed_dict)
					gt_min_pose = np.squeeze(np.take_along_axis(gt_pose, np.expand_dims(min_pose_idx[:, 1:], axis=1), axis=1), axis=1)
					angle_diff = np.mean(angle_diff_batch(gt_min_pose[:, 3:], pred_aa, aa=True, degree=True))

					for jj in range(pc_o.shape[0]):
						one_result_file_name = batch_dict['result_file_name'][jj]
						if not (one_result_file_name in eval_result_dict):
							eval_result_dict[one_result_file_name] = []
						eval_result_dict[one_result_file_name].append(
							{
								'z': z[jj].tolist(),
								'pred_transl': pred_transl[jj].tolist(),
								'pred_aa': pred_aa[jj].tolist(),
								'loss_transl': float(loss_transl_val[jj]),
								'loss_aa': float(loss_aa_val[jj])
							}
						)
					loss_transl_val = np.mean(loss_transl_val)
					loss_aa_val = np.mean(loss_aa_val)
					loss_transl_sum_test += loss_transl_val
					loss_aa_sum_test += loss_aa_val
					angle_diff_sum_test += angle_diff
					total_ct_test += 1
				print('epoch {} iter {}/{} loss_transl {:.4f},{:.4f} loss_aa {:.4f},{:.4f} angle diff {:.4f}'.format(epoch_i, i, epoch_iter, loss_transl_val, np.sqrt(loss_transl_val), loss_aa_val, np.sqrt(loss_aa_val), angle_diff))
			
			save_json(eval_file_dir, eval_result_dict)

			loss_transl_test = loss_transl_sum_test / total_ct_test 
			loss_aa_test = loss_aa_sum_test / total_ct_test    
			angle_diff_test = angle_diff_sum_test / total_ct_test
			print('epoch {} loss_transl_test {} loss_aa_test {} angle_diff_test {}'.format(epoch_i, loss_transl_test, loss_aa_test, angle_diff_test))

			if can_write:
				writer.add_scalar('test_epoch/loss_transl', loss_transl_test, total_ct)
				writer.add_scalar('test_epoch/loss_transl_sqrt', np.sqrt(loss_transl_test), total_ct)
				writer.add_scalar('test_epoch/loss_aa', loss_aa_test, total_ct)
				writer.add_scalar('test_epoch/loss_aa_sqrt', np.sqrt(loss_aa_test), total_ct)
				writer.add_scalar('test_epoch/angle_diff', angle_diff_test, total_ct)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")

	parser.add_argument('--model_name', default='s1_model')
	parser.add_argument('--comment', default='')
	parser.add_argument('--exp_name', default='exp_s1')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--log_freq', type=int, default=10)

	parser.add_argument('--train_list', default='train_list')
	parser.add_argument('--test_list', default='test_list')
	parser.add_argument('--restrict_object_cat', default='')

	parser.add_argument('--run_test', action='store_true')
	parser.add_argument('--no_save', action='store_true')
	parser.add_argument('--overfit', action='store_true')
	parser.add_argument('--restore_model_name', default='s1_model')
	parser.add_argument('--restore_model_epoch', type=int, default=40500)
	parser.add_argument('--max_epochs', type=int, default=10000)
	parser.add_argument('--eval_epoch_freq', type=int, default=2)
	parser.add_argument('--eval_sample_n', type=int, default=5)
	parser.add_argument('--model_save_freq', type=int, default=300)
	parser.add_argument('--no_eval', action='store_true')

	parser.add_argument('--data_more_pose', action='store_true')

	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--z_dim', type=int, default=32)
	parser.add_argument('--learning_rate', type=float, default=5e-4)

	parser.add_argument('--n_z_sample', type=int, default=15)
	parser.add_argument('--loss_transl_const', default=1000, type=float)

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
	if not args.run_test:
		train_set = MyDataset(args.home_dir_data, train_list_dir, is_train=True, use_partial_pc=True, args=args)
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
							num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict)
	
	test_set = MyDataset(args.home_dir_data, test_list_dir, is_train=False, use_partial_pc=True, args=args)
	test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
									num_workers=6, collate_fn=MyDataset.pad_collate_fn_for_dict)
	if not args.run_test:
		print('len of train {} len of test {}'.format(len(train_set), len(test_set)))
	else:
		print('len of train {} len of test {}'.format(len(test_set), len(test_set)))

	if not args.run_test:
		train(args, train_loader, test_loader, writer, result_folder, file_name)
	else:
		train(args, test_loader, test_loader, writer, result_folder, file_name)





