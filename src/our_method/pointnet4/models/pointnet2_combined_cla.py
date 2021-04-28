import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
print("BASE_DIR",BASE_DIR)
import tensorflow as tf
import numpy as np
from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_util

def get_model(point_cloud, l0_points, num_class, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    end_points['l0_xyz'] = l0_xyz

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.01, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.02, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.04, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.08, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    l5_xyz, l5_points, l5_indices = pointnet_sa_module(l4_xyz, l4_points, npoint=1, radius=0.20, nsample=16, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer5')

    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [1024,1024,512], is_training, bn_decay, scope='fa_layer0',bn=True)
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512,512,256], is_training, bn_decay, scope='fa_layer1',bn=True)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256,128], is_training, bn_decay, scope='fa_layer2',bn=True)
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128,128,128], is_training, bn_decay, scope='fa_layer3',bn=True)
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,64], is_training, bn_decay, scope='fa_layer4',bn=True)
  
    # FC layers
    net = l0_points
    end_points['feats'] = net 
    l0_points = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=False, is_training=is_training, scope='fc1_3', bn_decay=bn_decay)
    l0_points = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=False, is_training=is_training, scope='fc1_4', bn_decay=bn_decay)
    cla = tf_util.conv1d(l0_points, num_class, 1, padding='VALID', bn=False, is_training=is_training, scope='fc3_1', activation_fn=None)

    end_points['cla'] = cla
    return end_points

