import numpy as np
# import pybullet as p
import math
from scipy.spatial.transform import Rotation 
EPS = 1e-8

def normalize_vector(v):
    v = v / (EPS + np.linalg.norm(v))
    return v
def rpy_rotmat(rpy):
    rotmat = np.zeros((3,3))
    roll   = rpy[0]
    pitch  = rpy[1]
    yaw    = rpy[2]
    rotmat[0,0] = np.cos(yaw) * np.cos(pitch)
    rotmat[0,1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    rotmat[0,2] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    rotmat[1,0] = np.sin(yaw) * np.cos(pitch)
    rotmat[1,1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    rotmat[1,2] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    rotmat[2,0] = - np.sin(pitch)
    rotmat[2,1] = np.cos(pitch) * np.sin(roll)
    rotmat[2,2] = np.cos(pitch) * np.cos(roll)
    return rotmat

def local_coord_to_global(local_pos, local_origin_world_pos, local_origin_world_rpy):
    rotmat = rpy_rotmat(local_origin_world_rpy)
    T = np.zeros((4, 4))
    T[:3, :3] = rotmat
    T[:, 3] = np.append(local_origin_world_pos, [1], axis=0)

    local_pos_after = np.matmul(T, np.append(local_pos, [1]))

    assert local_pos_after[3] == 1
    return local_pos_after[:3] 

# def local_coord_to_global_bl_unit_vec(local_pos, object_id, link_id=0):
#     local_origin_world_pos = [0, 0, 0]
#     local_origin_world_rpy = p.getEulerFromQuaternion(p.getLinkState(object_id, link_id)[1])

#     normal = local_coord_to_global(local_pos, local_origin_world_pos, local_origin_world_rpy)
#     normal = normalize_vector(normal)
#     return normal

# def local_coord_to_global_bl(local_pos, object_id, link_id=0, obj_scaling=1):
#     if p.getLinkState(object_id, link_id) is None:
#         pos_rpy = p.getBasePositionAndOrientation(object_id)
#     else:
#         pos_rpy = p.getLinkState(object_id, link_id)
#     local_origin_world_pos = pos_rpy[0]
#     local_origin_world_rpy = p.getEulerFromQuaternion(pos_rpy[1])
#     print('origin', local_origin_world_pos, local_origin_world_rpy)
#     return local_coord_to_global(local_pos * obj_scaling, local_origin_world_pos, local_origin_world_rpy)


def filter_inside_bb(points, lower_corner, upper_corner):
    mask = np.sum(points >= lower_corner, axis=1) * np.sum(points <= upper_corner, axis=1) == lower_corner.shape[0]**2
    return mask
def filter_outside_bb(points, lower_corner, upper_corner):
    mask = np.sum(points >= lower_corner, axis=1) * np.sum(points <= upper_corner, axis=1) < lower_corner.shape[0]**2
    return mask

friction_coef = 0.3

def two_contacts_fc_Nguyen(p1, n1, p2, n2):

    vector_between_contacts = np.array(p1-p2)
    vector_between_contacts = vector_between_contacts/(EPS + np.linalg.norm(vector_between_contacts))
    cosine1 = np.abs(np.sum(vector_between_contacts * n1))
    cosine2 = np.abs(np.sum(vector_between_contacts * n2))
    thre = math.cos(math.atan(friction_coef))
    cosine = min(cosine1,cosine2)
    # print(cosine)
    if cosine > thre:
        return True
    else:
        return False

def two_points_normal_feasible(p1, n1, p2, n2):
    # check if n1 and n2 point opposite (angle > 120 deg)
    cosine = np.sum(n1 * n2)
    if_opposite = cosine < -0.5
    if_colinear = two_contacts_fc_Nguyen(p1, n1, p2, n2)

    return if_opposite and if_colinear

def interpolate(x, y, coef):
    assert coef <= 1 and coef >= 0
    return np.array(x) * (1 - coef) + np.array(y) * coef

def sample_points_in_sphere_uniform(n, radius=1, center=None):
    phi = np.random.uniform(0, 2*np.pi, size=[n])
    costheta = np.random.uniform(-1,1, size=[n])
    u = np.random.uniform(0,1, size=[n])

    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    ret = np.zeros((n, 3))

    ret[:, 0] =  r * np.sin( theta) * np.cos( phi )
    ret[:, 1] = r * np.sin( theta) * np.sin( phi )
    ret[:, 2] = r * np.cos( theta )

    if center is not None:
        ret += center

    return ret

def sample_points_on_sphere_uniform(n, center=None):
    phi = np.random.uniform(0, 2*np.pi, size=[n])
    costheta = np.random.uniform(-1,1, size=[n])

    theta = np.arccos(costheta)
    r = 1.

    ret = np.zeros((n, 3))

    ret[:, 0] =  r * np.sin( theta) * np.cos( phi )
    ret[:, 1] = r * np.sin( theta) * np.sin( phi )
    ret[:, 2] = r * np.cos( theta )

    if center is not None:
        ret += center

    return ret
    
def cartesian_product(*arrays, dtype=np.float32):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def sample_points_in_bb_uniform(n, bb_low, bb_high):
    sample_x = np.random.uniform(bb_low[0], bb_high[0], size=n)
    sample_y = np.random.uniform(bb_low[1], bb_high[1], size=n)
    sample_z = np.random.uniform(bb_low[2], bb_high[2], size=n)

    return np.stack([sample_x, sample_y, sample_z], axis=1)

def sample_quat_uniform(n):
    return Rotation.random(n).as_quat()

def local_pos_to_global(local_pos, local_origin_world_pos, scaling, local_origin_world_quat=None, p=None):
    if local_origin_world_quat is None:
        return local_origin_world_pos + scaling * local_pos
    else:
        rotmat = np.array(p.getMatrixFromQuaternion(local_origin_world_quat)).reshape(3, 3)
        return local_origin_world_pos + scaling * np.matmul(rotmat, local_pos)

def global_pos_to_local(global_pos, local_origin_world_pos, scaling, local_origin_world_quat=None, p=None):
    if local_origin_world_quat is None:
        return (global_pos - local_origin_world_pos) / scaling
    else:
        inv_pos, inv_quat = p.invertTransform(local_origin_world_pos, local_origin_world_quat)
        rotmat = np.array(p.getMatrixFromQuaternion(inv_quat)).reshape(3, 3)
        return inv_pos + scaling * np.matmul(rotmat, global_pos)

def is_overlapping_1d(box1, box2):
    return box1[1] >= box2[0] and box2[1] >= box1[0]
    
def is_overlapping_3d(box1, box2):		
    return is_overlapping_1d(box1[:, 0], box2[:, 0]) \
        and is_overlapping_1d(box1[:, 1], box2[:, 1]) \
        and is_overlapping_1d(box1[:, 2], box2[:, 2])
        
def get_quat_error(quat_1, quat_2):
    """
    Find the distance between two quaternions (w, x, y, z)
    :param quat_1: the first quaternion
    :param quat_2: the second quaternion
    :return: distance between the two quats
    """
    ori_err = min(np.linalg.norm(quat_1 - quat_2),np.linalg.norm(quat_1 + quat_2))
    ori_err = ori_err / math.sqrt(2)
    return ori_err

def get_quat_arr_error(quat_arr, quat):
    ori_err = np.minimum(np.linalg.norm(quat_arr - quat, axis=-1), np.linalg.norm(quat_arr + quat, axis=-1))
    ori_err = ori_err / math.sqrt(2)
    return np.mean(ori_err)

def mean_quat_error(obj_pos_quat_seq):
    return get_quat_arr_error(obj_pos_quat_seq[:, 3:], obj_pos_quat_seq[0, 3:])


def get_angle(vector_1, vector_2, degree=True):
    unit_vector_1 = vector_1 / (np.linalg.norm(vector_1))
    unit_vector_2 = vector_2 / (np.linalg.norm(vector_2))
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if dot_product > 1:
        dot_product = 1.

    angle = np.arccos(dot_product) 
    if degree:
        angle = angle / 3.14159 * 180.
        if angle < 0:
            angle += 360.
    return angle 

def get_angle_batch(v1, v2, degree=True):
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        return get_angle(v1, v2, degree=degree)

    if len(v1.shape) == 1:
        n_p = v2.shape[0]
        v1 = np.expand_dims(v1, axis=0)
        v1 = np.repeat(v1, n_p, axis=0)

    if len(v2.shape) == 1:        
        n_p = v1.shape[0]
        v2 = np.expand_dims(v2, axis=0)
        v2 = np.repeat(v2, n_p, axis=0)
    
    v1_norms = np.linalg.norm(v1, axis=1, keepdims=True)
    v1_norms[v1_norms == 0] = 1
    unit_v1 = v1 / v1_norms

    v2_norms = np.linalg.norm(v2, axis=1, keepdims=True)
    v2_norms[v2_norms == 0] = 1
    unit_v2 = v2 / v2_norms

    dot_product = np.sum(unit_v1 * unit_v2, axis=-1)
    dot_product[dot_product > 1] = 1
    angle = np.arccos(dot_product) 

    if degree:
        angle = angle / 3.14159 * 180.
    return angle

import pybullet
def apply_transform_to_pc(pc, pose):
    rotmat = np.array(pybullet.getMatrixFromQuaternion(pose[3:])).reshape(3, 3)
    return pose[:3] + np.matmul(rotmat, pc.transpose()).transpose()

def apply_transform_to_pc_with_n(pc, pose):
    out_pc = np.zeros_like(pc)
    rotmat = np.array(pybullet.getMatrixFromQuaternion(pose[3:])).reshape(3, 3)
    out_pc[:, :3] = pose[:3] + np.matmul(rotmat, pc[:, :3].transpose()).transpose()
    out_pc[:, 3:] = np.matmul(rotmat, pc[:, 3:].transpose()).transpose()
    return out_pc


def make_wall(hook_aabb, hook_world_pos, p, wall_size=None):

    if wall_size is None:
        wall_y_length = hook_aabb[1][1] - hook_aabb[0][1]
        wall_z_length = hook_aabb[1][2] - hook_aabb[0][2] 
    else:
        wall_y_length = wall_size
        wall_z_length = wall_size
        
    wall_thickness = 0.005
    wall_v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness/2., wall_y_length/2., wall_z_length/2.] )
    wall_c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2., wall_y_length/2., wall_z_length/2.])

    wall_world_pos = np.copy(hook_world_pos) * 1.
    wall_world_pos[0] += wall_thickness/2. + (hook_aabb[1][0] - hook_aabb[0][0])/2.
    wall_bullet_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_c_id, baseVisualShapeIndex=wall_v_id, basePosition=wall_world_pos)
    return wall_bullet_id

def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

def drawAABB(aabb, p):
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 0, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [0, 1, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [0, 0, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

if __name__ == '__main__':
    # points = np.array([[1, 1, 1], [2, 4, 2]])
    # lower_corner = np.array([0, 0, 0])
    # upper_corner = np.array([3, 3, 3])
    # print(filter_within_bb(points, lower_corner, upper_corner))

    # print(cartesian_product([1, 2], [3, 4], [5, 6]).shape)

    # local_pos = [1.2, -134, 23]
    # local_origin_world_pos = [2130, 40, -1001]
    # scaling = 1.
    # local_origin_world_quat = [0, 0, 0, 1]
    # local_origin_world_quat = [0.146, 0.354, 0.354, 0.854]

    # global_pos = local_pos_to_global(local_pos, local_origin_world_pos, 1., local_origin_world_quat) 
    # local_pos = global_pos_to_local(global_pos, local_origin_world_pos, 1., local_origin_world_quat)
    # global_pos_2 = local_coord_to_global(local_pos, local_origin_world_pos, p.getEulerFromQuaternion(local_origin_world_quat))
    # print(global_pos, local_pos)
    # print(global_pos_2)

    # q1 = np.array([0.7, 0, 0.7, 0.2])
    # q1_arr = np.array([[0.7, 0, 0.7, 0.2], [0.7, 0, 0.7, 0.2], [0.7, 0, 0.7, 0.2]])
    # q2 = np.array([0, 0, 0, 1])
    # print(get_quat_error(q1, q2))
    # print(get_quat_arr_error(q1_arr, q2))

    v1 = np.array([0, 1, 2])
    v1_b = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    v2 = np.array([0, 1, 2])

    print(get_angle(v1, v2))
    print(get_angle_batch(v1_b, v2))
    