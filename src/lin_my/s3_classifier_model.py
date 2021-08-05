import tensorflow as tf
import numpy as np
import tflearn
import sys
import os
import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET2_DIR = os.path.join(BASE_DIR,'pointnet4','models')
sys.path.insert(0,POINTNET2_DIR)

from pointnet2_combined_cla_all import get_model as enc_model

def placeholder_inputs(batch_size, num_point, with_normal, args):
    pc_combined_pl = tf.placeholder(tf.float32, shape=(None, num_point * 2, 4), name='pc_combined_normal')
    if with_normal:
        pc_combined_normal_pl = tf.placeholder(tf.float32, shape=(None, num_point * 2, 3), name='pc_combined_normal')
    gt_succ_label_pl = tf.placeholder(tf.float32, [None, ], name='gt_succ_label')
    return pc_combined_pl, gt_succ_label_pl


def get_model(pc_combined, pc_combined_normal=None):
    end_points = {}
    with tf.variable_scope('s3_classifier'):
        with tf.variable_scope('pointnet2'):
            if not pc_combined_normal is None:
                pc_combined_input_feat = tf.concat([pc_combined_normal, pc_combined[:, :, 3:]], axis=-1) 
            else:
                pc_combined_input_feat = pc_combined[:, :, 3:]  
            
            print('pc_combined_input_feat', pc_combined_input_feat)
            nets_end = enc_model(pc_combined[:, :, :3], pc_combined_input_feat, 2, is_training=True)

            pred_cla_score = nets_end['cla']
            print('pred_cla_score', pred_cla_score)
            end_points = nets_end
    return pred_cla_score, end_points

def get_loss(pred_cla_score, gt_succ_label, end_points):
    loss_cls_ce = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gt_succ_label,dtype=tf.int32),logits=pred_cla_score))
    return loss_cls_ce

if __name__ == '__main__':
    with tf.Graph().as_default():
        input_pc_combined = tf.zeros((32,4096,4))
        input_pc_combined_normal = tf.zeros((32,4096,3))
        get_model(input_pc_combined)
        