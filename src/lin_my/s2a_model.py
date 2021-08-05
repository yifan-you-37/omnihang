import tensorflow as tf
import numpy as np
import tflearn
import sys
import os
import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET2_DIR = os.path.join(BASE_DIR,'pointnet4','models')
sys.path.insert(0,POINTNET2_DIR)

from pointnet2_combined_cla import get_model as enc_model

def placeholder_inputs(batch_size, num_point, args):
	pc_combined_pl = tf.placeholder(tf.float32, shape=(None, num_point * 2, 4), name='pc_combined')
	gt_cp_score_o_pl = tf.placeholder(tf.float32, shape=(None, num_point), name='gt_cp_score_o')
	gt_cp_score_h_pl = tf.placeholder(tf.float32, shape=(None, num_point), name='gt_cp_score_h')
	non_nan_mask_pl = tf.placeholder(tf.bool, shape=(None, ), name='non_nan_mask')
	return pc_combined_pl, gt_cp_score_o_pl, gt_cp_score_h_pl, non_nan_mask_pl

def get_model(pc_combined, num_point, no_softmax=False):
	end_points = {}
	with tf.variable_scope('s2a_cp_prediction'):
		nets_end = enc_model(pc_combined[:, :, :3], pc_combined[:, :, 3:4], num_class=2, is_training=True)
		pred_cp_score_combined = nets_end['cla']

		if not no_softmax:
			pred_cp_score_combined = tf.nn.softmax(pred_cp_score_combined, axis=-1)
		pred_cp_score_o = pred_cp_score_combined[:, :num_point, 1]

		# pred_cp_score_o_max = tf.reduce_max(pred_cp_score_o, axis=-1, keepdims=True)
		# pred_cp_score_o_min = tf.reduce_min(pred_cp_score_o, axis=-1, keepdims=True)
		# print('pred_cp_score_o_max', pred_cp_score_o_max)
		# pred_cp_score_o_normalized = (pred_cp_score_o - pred_cp_score_o_min) / (pred_cp_score_o_max - pred_cp_score_o_min)

		pred_cp_score_h = pred_cp_score_combined[:, num_point:, 1]
		# pred_cp_score_h_max = tf.reduce_max(pred_cp_score_o, axis=-1, keepdims=True)
		# pred_cp_score_h_min = tf.reduce_min(pred_cp_score_o, axis=-1, keepdims=True)
		# print('pred_cp_score_h_max', pred_cp_score_h_max)
		# pred_cp_score_h_normalized = (pred_cp_score_o - pred_cp_score_h_min) / (pred_cp_score_h_max - pred_cp_score_h_min)
		print('pred_cp_score_o', pred_cp_score_o)
		print('pred_cp_score_h', pred_cp_score_h)
		end_points['pc_combined_feat'] = nets_end['feats']

	return pred_cp_score_o, pred_cp_score_h, end_points

def get_loss(pred_cp_score_o, pred_cp_score_h, gt_cp_score_o, gt_cp_score_h, end_points):
	loss_huber_o_tf = tf.losses.huber_loss(gt_cp_score_o, pred_cp_score_o, reduction=tf.losses.Reduction.NONE, delta=1.)
	loss_huber_h_tf = tf.losses.huber_loss(gt_cp_score_h, pred_cp_score_h, reduction=tf.losses.Reduction.NONE, delta=1.)
	
	# loss_o_tf = ((gt_cp_score_o - pred_cp_score_o) ** 2) * 0.5
	# loss_h_tf = ((gt_cp_score_h - pred_cp_score_h) ** 2) * 0.5 + 1
	
	# loss_huber_h_tf = tf.where(tf.is_nan(loss_huber_h_tf), tf.ones(loss_huber_h_tf) * -1., loss_huber_h_tf)
	# loss_huber_h_tf = tf.where(tf.is_nan(loss_huber_o_tf), tf.ones(loss_huber_h_tf) * -1., loss_huber_h_tf)
	# non_nan_mask = tf.math.not_equal(loss_huber_h_tf, tf.constant(-1, dtype=tf.float32))
	
	# loss_huber_o_tf = tf.boolean_mask(loss_huber_o_tf, non_nan_mask)
	# loss_huber_h_tf = tf.boolean_mask(loss_huber_h_tf, non_nan_mask)

	print('loss huber o', loss_huber_o_tf)
	print('loss huber h', loss_huber_h_tf)

	# with tf.control_dependencies([tf.assert_equal(loss_o_tf, loss_huber_o_tf), tf.assert_equal(loss_h_tf, loss_huber_h_tf)]):
	loss_huber_o_tf = tf.reduce_mean(loss_huber_o_tf, axis=-1)
	loss_huber_h_tf = tf.reduce_mean(loss_huber_h_tf, axis=-1)
	
	
	return loss_huber_o_tf, loss_huber_h_tf

if __name__ == '__main__':
	with tf.Graph().as_default():
		input_pc_combined = tf.zeros((32,4096 * 2,4))
		get_model(input_pc_combined, 4096)
		