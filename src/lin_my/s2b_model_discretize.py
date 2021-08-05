import tensorflow as tf
import numpy as np
import tflearn
import sys
import os
import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET2_DIR = os.path.join(BASE_DIR,'pointnet4','models')
sys.path.insert(0,POINTNET2_DIR)

from point_quality import two_point_quality
from pointnet2_combined_cla import get_model as enc_model
from tf_helper import *

def placeholder_inputs(batch_size, num_point, args):
	gt_cp_corr_label_pl = tf.placeholder(tf.int32,[None, args.top_k_o * args.top_k_h], 'gt_cp_corr_label_pl')
	
	return gt_cp_corr_label_pl

def get_model(pc_combined, pred_cp_score_o, pred_cp_score_h, num_point, batch_size):
	end_points = {}
	k_o = 128
	k_h = 128
	with tf.variable_scope('s2b_cp_correspondence'):
		with tf.variable_scope('encoder'):
			nets_end = enc_model(pc_combined[:, :, :3], pc_combined[:, :, 3:4], num_class=2, is_training=True)
			enc_feat = nets_end['feats']
			print('enc_feat', enc_feat)

		with tf.variable_scope('corr_conv'):
			_, pred_cp_top_k_idx_o = tf.nn.top_k(pred_cp_score_o, k=k_o, sorted=True)
			_, pred_cp_top_k_idx_h = tf.nn.top_k(pred_cp_score_h, k=k_h, sorted=True)

			end_points['pred_cp_top_k_idx_o'] = pred_cp_top_k_idx_o
			end_points['pred_cp_top_k_idx_h'] = pred_cp_top_k_idx_h

			pred_cp_feat_o = enc_feat[:, :num_point, :]
			pred_cp_feat_h = enc_feat[:, num_point:, :]

			print('pred_cp_feat_o', pred_cp_feat_o)
			print('pred_cp_feat_h', pred_cp_feat_h)

			top_k_feat_list_o = []
			top_k_feat_list_h = []
			for bi in range(batch_size):
				top_k_feat_list_o.append(tf.gather(pred_cp_feat_o[bi], pred_cp_top_k_idx_o[bi]))
				top_k_feat_list_h.append(tf.gather(pred_cp_feat_h[bi], pred_cp_top_k_idx_h[bi]))
			
			top_k_feat_o = tf.convert_to_tensor(top_k_feat_list_o)
			top_k_feat_h = tf.convert_to_tensor(top_k_feat_list_h)
			
			pred_cp_corr_logit, _ = two_point_quality(top_k_feat_o, top_k_feat_h, batch_size, k_o, k_h)
			print('pred_cp_corr_logit', pred_cp_corr_logit)
			pred_cp_corr = tf.nn.softmax(pred_cp_corr_logit, axis=-1)[:, :, 1]
			print('pred_cp_corr_after_softmax', pred_cp_corr)
			# pred_cp_corr_normalized = normalize_tensor(pred_cp_corr)
			# print('pred_cp_corr_normalized', pred_cp_corr_normalized)
			end_points['pred_cp_corr_logit'] = pred_cp_corr_logit
		
	return pred_cp_corr, end_points

def get_loss(pred_cp_corr_logit, gt_cp_corr_label, end_points):
	pred_cp_corr_score = pred_cp_corr_logit[:, :, 1]
	denominator = tf.reduce_sum(tf.exp(pred_cp_corr_score), axis=1, keep_dims=True)
	loss_listnet = -tf.reduce_mean(tf.cast(gt_cp_corr_label,dtype=tf.float32) * tf.log(tf.exp(pred_cp_corr_score) / denominator), axis=-1)
	loss_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gt_cp_corr_label,dtype=tf.int32), logits=pred_cp_corr_logit), axis=-1) 
	
	print('loss_listnet', loss_listnet)
	print('loss_ce', loss_ce)
	return loss_listnet, loss_ce

if __name__ == '__main__':
	with tf.Graph().as_default():
		input_pc_combined = tf.zeros((32,4096 * 2,4))
		input_pred_cp_score_o = tf.zeros((32, 4096))
		input_pred_cp_score_h = tf.zeros((32, 4096))
		input_s2a_feat = tf.zeros((32, 4096, 64))

		get_model(input_pred_cp_score_o, input_pred_cp_score_h, input_s2a_feat, 4096, 32)
		