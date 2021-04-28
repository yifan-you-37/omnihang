import numpy as np
import fcl

from data_helper import * 
from coord_helper import *
# from train_helper import *
from rotation_lib import *

from obj_loader import obj_loader

def fcl_load_obj(obj_dir, scaling=1.):
    obj_v, obj_f, _ = obj_loader(obj_dir)
    obj_v *= scaling

    m = fcl.BVHModel()
    m.beginModel(len(obj_v), len(obj_f))
    m.addSubModel(obj_v, obj_f)
    m.endModel()
    return m

def fcl_half_load_obj(obj_dir, scaling=1.):
    obj_v, obj_f, _ = obj_loader(obj_dir)
    obj_v *= scaling
    return {
        'obj_v': obj_v,
        'obj_f': obj_f
    }
def fcl_half_load_urdf(urdf_dir):
    _, scaling = get_name_scale_from_urdf(urdf_dir)
    obj_dir = get_obj_dir_from_urdf(urdf_dir)
    return fcl_half_load_obj(obj_dir, scaling)
    
def fcl_model_to_fcl(obj_v, obj_f):
    m = fcl.BVHModel()
    m.beginModel(len(obj_v), len(obj_f))
    m.addSubModel(obj_v, obj_f)
    m.endModel()
    return m

def fcl_load_urdf(urdf_dir):
    _, scaling = get_name_scale_from_urdf(urdf_dir)
    obj_dir = get_obj_dir_from_urdf(urdf_dir)
    return fcl_load_obj(obj_dir, scaling)
    
def fcl_get_dist(hook_model, object_model, pose_transl, pose_quat, aa=False, obj_dir=False, urdf=False, hook_scaling=1., object_scaling=1.):    
    if aa:
        pose_quat = quaternion_from_angle_axis(pose_quat) 
    if obj_dir:
        hook_obj_dir = hook_model
        object_obj_dir = object_model

    if urdf:
        _, hook_scaling = get_name_scale_from_urdf(hook_model)
        _, object_scaling = get_name_scale_from_urdf(object_model)
        hook_obj_dir = get_obj_dir_from_urdf(hook_model)
        object_obj_dir = get_obj_dir_from_urdf(object_model)
        # object_obj_dir = object_obj_dir[:-4] + '_wt.obj'
        # print(object_obj_dir)
    if obj_dir or urdf:
        hook_model = fcl_load_obj(hook_obj_dir, hook_scaling)
        object_model = fcl_load_obj(object_obj_dir, object_scaling)

    # req = fcl.CollisionRequest()
    # res = fcl.CollisionResult()
    # n_collide = fcl.collide(
    #     fcl.CollisionObject(hook_model, fcl.Transform()),
    #     fcl.CollisionObject(object_model, fcl.Transform(quat2mat(pose_quat), pose_transl)),
    #     req,
    #     res
    # )
    dist = fcl.distance(
        fcl.CollisionObject(hook_model, fcl.Transform()),
        fcl.CollisionObject(object_model, fcl.Transform(quat2mat(pose_quat), pose_transl)),
        fcl.DistanceRequest(),
        fcl.DistanceResult()
    )
    return dist
    # print(res.nearest_points, type(dist))
    # return n_collide

