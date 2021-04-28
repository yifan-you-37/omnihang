import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from math import ceil,trunc,floor,sin,cos,atan,acos,sqrt

EPS = 1e-6

def angle_axis_from_quaternion(quater):
  angle = 2 * acos(quater[3])
  axis = quater[:3]/(sin(angle/2)+EPS)
  return angle * axis

def angle_axis_from_quaternion_batch(quarter):
  angle = 2 * np.arccos(quarter[:, 3])[:, np.newaxis]
  axis = quarter[:, :3] / (np.sin(angle/2) + EPS)
  return angle * axis

def quaternion_from_angle_axis(aa):
  angle = np.linalg.norm(aa)
  axis = aa / (angle + EPS)
  q0 = cos(angle/2)
  # print(axis,angle,sin(angle/2))
  qx,qy,qz = axis * sin(angle/2)
  return np.array([qx,qy,qz,q0])

def angle_diff(q1, q2, degree=False):
  # https://math.stackexchange.com/questions/90081/quaternion-distance
  # q1_inv = R.from_quat(q1).inv().as_quat()
  # print('inv', q1, q1_inv)
  # res = q2 * q1_inv
  tmp = 2 * np.sum(q1 * q2)**2 - 1 + EPS
  if tmp > 1:
    tmp = 1
  diff_rad = np.arccos(tmp) 
  if degree:
    return diff_rad / np.pi * 180.
  return diff_rad

def angle_diff_batch(q1, q2, degree=False, aa=False):
  if len(q1.shape) == 1:
    q1 = q1[np.newaxis, :]
  if len(q2.shape) == 1:
    q2 = q2[np.newaxis, :]
  if aa:
    q1 = quaternion_from_angle_axis_batch(q1)
    q2 = quaternion_from_angle_axis_batch(q2)
  
  tmp = 2 * np.sum(q1 * q2, axis=-1)**2 - 1 + EPS
  tmp[tmp > 1] = 1
  diff_rad = np.arccos(tmp)
  if degree:
    return diff_rad / np.pi * 180.
  return diff_rad

def quaternion_from_angle_axis_batch(aa):
  angle = np.linalg.norm(aa, axis=-1, keepdims=True)
  axis = aa/(angle + EPS)
  q0 = np.cos(angle/2)
  tmp =  axis * np.sin(angle/2)
  return np.append(tmp, q0, axis=-1)

MAX_FLOAT = np.maximum_sctype(np.float)
FLOAT_EPS = np.finfo(np.float).eps

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    x, y, z, w = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
            
from scipy.spatial.transform import Rotation 
def pose_to_mat_4(pose):
    transl = pose[:3]
    rotmat = quat2mat(pose[3:])
    T = np.zeros((4, 4))
    T[:3, :3] = rotmat
    T[:, 3] = np.append(transl, [1], axis=0)
    return T
def mat_4_to_pose(pose_mat):
    pose = np.zeros((7))
    pose[:3] = pose_mat[:3, 3]
    rotmat = pose_mat[:3, :3]
    pose[3:] = Rotation.from_matrix(rotmat).as_quat()
    return pose

def is_aa(x):
  return x.shape[-1] == 3
def transform_pc_batch(pc, pos, quat, normal=False, aa=False):
    aa = is_aa(quat)
    if pc.shape[-1] == 6:
      normal = True
    if aa:
        quat = quaternion_from_angle_axis_batch(quat)
    new_pc = np.zeros_like(pc)
    rotmat = Rotation.from_quat(quat).as_matrix()
    rotmat = np.swapaxes(rotmat, -1, -2)
    new_pc[:, :, :3] = pc[:, :, :3] @ rotmat + pos[:, np.newaxis, :]
    if normal:
      new_pc[:, :, 3:] = pc[:, :, 3:] @ rotmat
    return new_pc

def transform_pc(pc, pos, quat, normal=False, aa=False):
  aa = is_aa(quat)
  if pc.shape[-1] == 6:
    normal = True
  if aa:
    quat = quaternion_from_angle_axis(quat)
  
  new_pc = np.zeros_like(pc) 
  new_pc[:, :3] = np.dot(pc[:, :3], Rotation.from_quat(quat).as_matrix().T) + pos

  if normal:
    new_pc[:, 3:] = np.dot(pc[:, 3:], Rotation.from_quat(quat).as_matrix().T)
  return new_pc

def create_pc_combined(pc_o, pc_h, pos, quat, normal=False, aa=False):
  aa = is_aa(quat)
  assert pc_o.shape[0] == 4096 and pc_h.shape[0] == 4096
  if not normal:
    pc_o = pc_o[:, :3]
    pc_h = pc_h[:, :3]

  num_point = 4096
  pc_o_end = transform_pc(pc_o, pos, quat, normal=normal, aa=aa)
  pc_o_end_w_label = np.append(pc_o_end, np.ones((num_point, 1)), axis=1)
  pc_h_w_label = np.append(pc_h, np.zeros((num_point, 1)), axis=1)
  pc_combined = np.append(pc_o_end_w_label, pc_h_w_label, axis=0)
  
  return pc_combined

def create_pc_combined_batch(pc_o, pc_h, pos=None, quat=None, normal=False, aa=False):
  if not (quat is None):
    aa = is_aa(quat)
  assert pc_o.shape[1] == 4096 and pc_h.shape[1] == 4096
  if not normal:
    pc_o = pc_o[:, :, :3]
    pc_h = pc_h[:, :, :3]

  num_point = 4096
  b_size = pc_o.shape[0]

  if (pos is None) and (quat is None):
    pc_o_end = pc_o
  else:
    pc_o_end = transform_pc_batch(pc_o, pos, quat, normal=normal, aa=aa)
  
  pc_o_end_w_label = np.append(pc_o_end, np.ones((b_size, num_point, 1)), axis=-1)
  pc_h_w_label = np.append(pc_h, np.zeros((b_size, num_point, 1)), axis=-1)
  pc_combined = np.append(pc_o_end_w_label, pc_h_w_label, axis=1)

  return pc_combined

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    aa = angle_axis_from_quaternion(Rotation.from_matrix(R).as_quat())
    # print(np.max(np.abs(transform_pc(A, t, aa) - B)))
    # return T, R, t
    return t, aa

def best_fit_transform_batch(pc_a, pc_b):
  b_size = pc_a.shape[0]
  assert np.array_equal(pc_a.shape, pc_b.shape)

  pred_transl = np.zeros((b_size, 3))
  pred_aa = np.zeros((b_size, 3))
  for ii in range(b_size):
    transl_tmp, aa_tmp = best_fit_transform(pc_a[ii], pc_b[ii])
    pred_transl[ii] = transl_tmp
    pred_aa[ii] = aa_tmp
  return pred_transl, pred_aa



# testing
def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

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

def get_pc_bb(pc):
  return np.array([np.min(pc, axis=0), np.max(pc, axis=0)])
# def test_best_fit():
#     import time
#     N = 10                                    # number of random points in the dataset
#     num_tests = 100                             # number of test iterations
#     dim = 3                                     # number of dimensions of the points
#     noise_sigma = .01                           # standard deviation error to be added
#     translation = .1                            # max translation of the test set
#     rotation = .1      
#     # Generate a random dataset
#     A = np.random.rand(N, dim)

#     total_time = 0

#     for i in range(num_tests):

#         B = np.copy(A)

#         # Translate
#         t = np.random.rand(dim)*translation
#         B += t

#         # Rotate
#         R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
#         B = np.dot(R, B.T).T

#         # Add noise
#         B += np.random.randn(N, dim) * noise_sigma

#         # Find best fit transform
#         start = time.time()
#         T, R1, t1 = best_fit_transform(B, A)
#         total_time += time.time() - start

#         # Make C a homogeneous representation of B
#         C = np.ones((N, 4))
#         C[:,0:3] = B

#         # Transform C
#         C = np.dot(T, C.T).T
#         # print(np.max(np.abs(C[:, 0:3] - (np.dot(B, R1.T) + t1))))
#         assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
#         assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
#         assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

#     print('best fit time: {:.3}'.format(total_time/num_tests))

#     return

def test_best_fit_my():
    import time
    N = 10                                    # number of random points in the dataset
    num_tests = 100                             # number of test iterations
    dim = 3                                     # number of dimensions of the points
    noise_sigma = 0                          # standard deviation error to be added
    translation = 2                            # max translation of the test set
    rotation = .8      
    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        pred_transl, pred_aa = best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        # C = np.ones((N, 4))
        # C[:,0:3] = B
        C = transform_pc(B, pred_transl, pred_aa)
        print(np.max(np.abs(C - A)))
        # # Transform C
        # C = np.dot(T, C.T).T
        # # print(np.max(np.abs(C[:, 0:3] - (np.dot(B, R1.T) + t1))))
        # assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        # assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        # assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses
        
    print('best fit time: {:.3}'.format(total_time/num_tests))

    b_size = 32
    num_points = 4096
    A = np.random.rand(b_size, num_points, 3) + 0.24
    gt_transl = np.random.rand(b_size, 3) 
    from scipy.spatial.transform import Rotation 
    gt_quat = Rotation.random(b_size).as_quat()
    gt_aa = angle_axis_from_quaternion_batch(gt_quat)
    B = transform_pc_batch(A, gt_transl, gt_quat) 
    pred_transl, pred_aa = best_fit_transform_batch(A, B)
    pred_quat = quaternion_from_angle_axis_batch(pred_aa)
    pred_B = transform_pc_batch(A, pred_transl, pred_quat)
    print('transl error', np.max(np.abs(gt_transl - pred_transl)))
    print('aa error', np.max(np.abs(angle_diff_batch(gt_quat, pred_quat, True))))
    print('pc error', np.max(np.abs(B - pred_B)))
    for i in range(b_size):
      print(get_quat_error(gt_quat[i], pred_quat[i]))
if __name__ == "__main__":
  quater = np.array([-0.4676878,-0.485222,-0.50507426,0.5391918])
  tmp = angle_axis_from_quaternion(quater)
  res = quaternion_from_angle_axis(tmp)

  # print('aa ori', tmp)
  # tmp = np.array([tmp,tmp] )
  # print(tmp.shape)
  # print('output', quaternion_from_angle_axis_batch(tmp).shape)
  # print(quaternion_from_angle_axis_batch(tmp))
  # print('aa batch', angle_axis_from_quaternion_batch(quaternion_from_angle_axis_batch(tmp)))
  # print(res)

  # import pybullet as p
  # a = p.getMatrixFromQuaternion(quater)
  # rotmat = quat2mat(quater)

  # print(np.allclose(a, rotmat.reshape(-1,)))

  # trans = np.array([1, 2, 3])
  # pc = np.random.random((1000, 3))
  # out_pc1 = np.dot(pc, rotmat.T) + trans
  # out_pc2 = (np.dot(rotmat, pc.T).T + trans)
  # print(np.allclose(out_pc1, out_pc2))

  # transl = np.random.rand(2, 3)
  # pc = np.random.rand(2, 4096, 3)

  # transl_2 = np.random.rand(3)
  # from scipy.spatial.transform import Rotation as R
  # quat = R.random(2).as_quat()
  
  # pc_1_after = transform_pc(pc[0], transl[0], quat[0])
  # pc_2_after = transform_pc(pc[1], transl[0], quat[0])
  # pc_after = transform_pc_batch(pc, transl, quat)

  # assert np.allclose(pc_1_after, pc_after[0])
  # assert np.allclose(pc_2_after, pc_after[1])
  print(test_best_fit_my())