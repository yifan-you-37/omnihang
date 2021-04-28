import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module
import tflearn

def get_model(point_cloud, l0_points, num_class, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    end_points['l0_xyz'] = l0_xyz
    print('point_cloud', point_cloud)
    print('l0_points', l0_points)
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.01, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.02, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.04, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.08, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    l5_xyz, l5_points, l5_indices = pointnet_sa_module(l4_xyz, l4_points, npoint=1, radius=0.20, nsample=16, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer5')

    # Fully connected layers
    # print('l5_points', l5_points)
    # net = tf.reshape(l5_points, [tf.shape(l5_points)[0], -1])
    net = tf.squeeze(l5_points, axis=1)

    net = tflearn.layers.core.fully_connected(net,512,activation=tf.nn.leaky_relu)
    net = tflearn.layers.core.fully_connected(net,256,activation=tf.nn.leaky_relu)
    net = tflearn.layers.core.fully_connected(net,2,activation=None)

    end_points['cla'] = net
    return end_points
