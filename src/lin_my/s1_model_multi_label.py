import tensorflow as tf
import numpy as np
import tflearn
import sys
import os
import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET2_DIR = os.path.join(BASE_DIR,'pointnet4','models')
sys.path.insert(0,POINTNET2_DIR)

from pointnet2_obj_enc import get_model as enc_model

def placeholder_inputs(batch_size, num_point, args):
    pc_o_pl = tf.placeholder(tf.float32, shape=(None, num_point, 3), name='pc_o')
    pc_h_pl = tf.placeholder(tf.float32, shape=(None, num_point, 3), name='pc_h')
    z_pl = tf.placeholder(tf.float32, [None, 1, args.z_dim], name='z')

    gt_obj_transl_pl = tf.placeholder(tf.float32, [None, None, 3], name='gt_transl')
    gt_obj_aa_pl = tf.placeholder(tf.float32, [None, None, 3], name='gt_aa')

    pose_mult_pl = tf.placeholder(tf.float32, [None, None], name='pose_mult')
    return pc_o_pl, pc_h_pl, z_pl, gt_obj_transl_pl, gt_obj_aa_pl, pose_mult_pl

def get_model(pc_o, pc_h, z):
    end_points = {}
    with tf.variable_scope('s1_encode'):
        with tf.variable_scope('encoder_o'):
            nets_end = enc_model(pc_o, is_training=True, bn_decay=None)
            enc_feat_o = nets_end['l6_points']
            print('enc_feat_o', enc_feat_o)

        with tf.variable_scope('encoder_h'):
            nets_end = enc_model(pc_h, is_training=True, bn_decay=None)
            enc_feat_h = nets_end['l6_points']
            print('enc_feat_h', enc_feat_h)
        
        with tf.variable_scope('encoder_z'):
            enc_feat_z = tflearn.layers.core.fully_connected(z, 512,activation=tf.nn.leaky_relu)
            enc_feat_z = tflearn.layers.core.fully_connected(enc_feat_z, 512, activation=tf.nn.leaky_relu)
            enc_feat_z = tf.expand_dims(enc_feat_z, axis=1)
            print('enc_feat_z', enc_feat_z)
        
        enc_feat_all = tf.concat([enc_feat_o, enc_feat_h, enc_feat_z], axis=-1)
        end_points['enc_feat_all'] = enc_feat_all

    with tf.variable_scope('s1_pose_prediction'):
        pose_feat = tf.squeeze(enc_feat_all,axis=1)
        pose_feat = tflearn.layers.core.fully_connected(pose_feat,512,activation=tf.nn.leaky_relu)
        
        pose_t = tflearn.layers.core.fully_connected(pose_feat,128,activation=tf.nn.leaky_relu)
        pose_t = tflearn.layers.core.fully_connected(pose_t,64,activation=tf.nn.leaky_relu)
        pose_t = tflearn.layers.core.fully_connected(pose_t,64,activation=tf.nn.leaky_relu)
        pose_t = tflearn.layers.core.fully_connected(pose_t,32,activation=tf.nn.leaky_relu)
        
        pose_transl = tflearn.layers.core.fully_connected(pose_t,3)
        
        pose_a = tflearn.layers.core.fully_connected(pose_feat,128,activation=tf.nn.leaky_relu)
        pose_a = tflearn.layers.core.fully_connected(pose_a,64,activation=tf.nn.leaky_relu)
        pose_a = tflearn.layers.core.fully_connected(pose_a,64,activation=tf.nn.leaky_relu)
        pose_a = tflearn.layers.core.fully_connected(pose_a,32,activation=tf.nn.leaky_relu)

        pose_aa = tflearn.layers.core.fully_connected(pose_a,3)

        print('pose_transl', pose_transl)
        print('pose_aa', pose_aa)

    return pose_transl, pose_aa, end_points

def get_loss(pred_transl, pred_aa, gt_transl, gt_aa, pose_mult, loss_transl_const=1., end_points=None):
    pred_transl = tf.expand_dims(pred_transl, axis=1)
    pred_aa = tf.expand_dims(pred_aa, axis=1)

    # loss_transl = tf.reduce_mean(tf.squared_difference(pred_transl, gt_transl))
    # loss_aa = tf.reduce_mean(tf.squared_difference(pred_aa, gt_aa))
    loss_all_transl = tf.reduce_mean(tf.squared_difference(pred_transl, gt_transl), axis=-1)
    loss_all_aa = tf.reduce_mean(tf.squared_difference(pred_aa, gt_aa), axis=-1)

    loss_all = loss_all_transl * loss_transl_const + loss_all_aa
    min_idx = tf.argmin(loss_all * pose_mult, axis=-1)
    min_idx = tf.stack([tf.range(tf.shape(min_idx)[0]), tf.dtypes.cast(min_idx, tf.int32)], axis=-1)
    loss_transl = tf.gather_nd(loss_all_transl, min_idx)
    loss_aa = tf.gather_nd(loss_all_aa, min_idx)
    # loss_transl = tf.gather(loss_all_transl, min_idx, axis=-1)
    # loss_aa = tf.gather(loss_all_aa, min_idx, axis=-1)
    return loss_transl, loss_aa, min_idx

if __name__ == '__main__':
    with tf.Graph().as_default():
        input_o = tf.zeros((32,4096,3))
        input_h = tf.zeros((32,4096,3))
        input_z = tf.zeros((32, 1, 32))
        get_model(input_o, input_h, input_z)
        