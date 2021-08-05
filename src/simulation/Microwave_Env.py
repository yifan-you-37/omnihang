import random
import os
import time
import sys
import numpy as np
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import pickle
import math
# import cv2

from math import sin,cos,acos

import robot
from matplotlib import pyplot as plt
import scipy.spatial.distance 
import scipy.ndimage
from scipy import ndimage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'utils'))
from coord_helper import * 
from data_helper import * 

def check_outside_tray(obj_pos, tray_bbox):
  diff = tray_bbox - obj_pos
  sign = np.sign(diff[0,:] * diff[1, :])[:2]
  return np.any(sign > 0) 


class RobotEnv():
  def __init__(self,
               worker_id,
               p_,
               objects,
               client_id,
               repo_dir=None, 
               use_robot=True,
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               maxSteps=20,
               dv=0.01,
               dt=0.001,
               blockRandom=0.01,
               cameraRandom=0,
               width=640,
               height=480,
               start_pos = [0.5, 0.3, 0.5],
               fixture_offset=np.zeros((3,)),
               isTest=False,
               is3D=False):
    self._timeStep = 1./240.
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = 20#maxSteps
    self._isDiscrete = False
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180 
    self._cam_pitch = -40
    self._dv = dv
    self.p = p_
    self.client_id = client_id
    self.objects = objects
    self.delta_t = dt
    self.p.setTimeStep(self.delta_t)
    self.fixture_offset = fixture_offset

    self.p.setPhysicsEngineParameter(enableConeFriction=1)
    self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
    self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
    self.p.setPhysicsEngineParameter(numSolverIterations=40)
    self.p.setPhysicsEngineParameter(numSubSteps=40)
    self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
    self.p.setPhysicsEngineParameter(enableFileCaching=0)

    self.p.setTimeStep(1 / 100.0)
    self.p.setGravity(0,0,-9.81)

    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._isTest = isTest
    self._wid = worker_id
    self.termination_flag = False
    self.success_flag = False

    self.start_pos = start_pos
    self.robot = robot
    
    if repo_dir is None:
      BASE_DIR = os.path.dirname(os.path.abspath(__file__))
      repo_dir = os.path.abspath(os.path.join(BASE_DIR, '..'))

    self.resource_dir = os.path.join(repo_dir, 'resource')
    self.texture_dir = os.path.join(self.resource_dir,"texture")
    self.cameraPose_dir = os.path.join(self.resource_dir,"cameraPose")

    self.urdf_dir = os.path.join(self.resource_dir,"urdf")
    self.plane_id = self.p.loadURDF(os.path.join(self.urdf_dir,"plane.urdf"),[0,0,0])
    self.env_textid = self.p.loadTexture(os.path.join(self.texture_dir,"texture1.jpg"))
    self.p.changeVisualShape(0,-1,textureUniqueId=self.env_textid)

    self._env_step = 0
    #### table initialization    
    self.table_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[0.3,0.5,0.15])
    self.table_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[0.3,0.5,0.15])
    mass = 0
    self.table_id = self.p.createMultiBody(mass,baseCollisionShapeIndex=self.table_c,baseVisualShapeIndex=self.table_v,basePosition=(0.5,0.0,0.2))
    self.table_color = [128/255.0,128/255.0,128/255.0,1.0]
    self.p.changeVisualShape(self.table_id,-1,rgbaColor=self.table_color)

    ####  robot initialization
    self.red_color = [0.9254901, 0.243137, 0.086274509,1.0]
    self.blue_color = [0.12156, 0.3804, 0.745, 1.0]
    self.yellow_color = [0.949, 0.878, 0.0392, 1.0]
    
    self.use_robot = use_robot
    if self.use_robot:
      self.robot = robot.Robot(pybullet_api=self.p,urdf_path=self.urdf_dir, start_pos=[0, 0, 0])
      self.p.changeVisualShape( self.robot.robotId, self.robot.gripper_right_tip_index, rgbaColor=self.blue_color,specularColor=[1.,1.,1.])

    self.init_obj()
    self.reset()

  def init_obj(self):

    # table_z = self.p.getAABB(self.table_id)[1][2]
    for obj in self.objects:
      init_pos = obj.init_pos
      init_pos[2] += table_z
      self.obj_id = self.p.loadURDF( obj.urdf_path, basePosition=init_pos, baseOrientation=obj.init_quat, globalScaling=obj.scaling,useFixedBase=True)
      self.p.changeVisualShape( self.obj_id, 1, rgbaColor=self.blue_color,specularColor=[1.,1.,1.])
      self.p.changeVisualShape( self.obj_id, 0, rgbaColor=self.yellow_color,specularColor=[1.,1.,1.])
      obj.client_id = self.client_id
      obj.obj_id_bl = self.obj_id

    obj_friction_ceof = 0.01
    # self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
    # self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
    # self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
    # self.p.changeDynamics(self.obj_id, -1, linearDamping=0.1)
    # self.p.changeDynamics(self.obj_id, -1, angularDamping=0.1)
    # self.p.changeDynamics(self.obj_id, -1, contactStiffness=300.0, contactDamping=0.1)

    # self.p.changeDynamics(self.obj_id, 1, lateralFriction=obj_friction_ceof)
    # self.p.changeDynamics(self.obj_id, 1, rollingFriction=obj_friction_ceof)
    # self.p.changeDynamics(self.obj_id, 1, spinningFriction=obj_friction_ceof)
    # self.p.changeDynamics(self.obj_id, 1, contactStiffness=300.0, contactDamping=0.1)


    # table_friction_ceof = 0.4
    # self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
    # self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
    # self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
    # self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)


  def obj_reset(self):
    # microwave
    for obj in self.objects:
      self.p.resetJointState(self.obj_id, 0, targetValue=0, targetVelocity=0.0)
      self.p.resetBasePositionAndOrientation(self.obj_id, obj.init_pos, obj.init_quat)


  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    if self.use_robot:
      self.robot.reset()
    self.obj_reset()

    # Set the camera settings.
    viewMatrix = np.loadtxt(os.path.join(self.cameraPose_dir, "handeye.txt"))
    cameraEyePosition = viewMatrix[:3, 3]
    cameraUpVector = viewMatrix[:3, 1] * -1.0
    cameraTargetPosition = viewMatrix[:3, 3] + viewMatrix[:3, 2] * 0.001
    self._view_matrix = self.p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)
    self._view_matrix_np = np.eye(4)
    self._view_matrix_np = np.array(self._view_matrix)
    self._view_matrix_np = self._view_matrix_np.reshape((4,4)).T
    self._view_matrix_inv = np.linalg.inv(self._view_matrix_np)
    self.cameraMatrix = np.load(os.path.join(self.cameraPose_dir, "cameraExPar.npy"))
    fov = 2 * math.atan(self._height / (2 * self.cameraMatrix[1, 1])) / math.pi * 180.0
    self.fov = fov
    aspect = float(self._width) / float(self._height)

    # fov = 60
    # aspect = 1.
    near = 0.02
    far = 4
    self._proj_matrix = self.p.computeProjectionMatrixFOV(fov, aspect, near, far)
    self.far = far
    self.near = near
    self._proj_matrix = np.array(self._proj_matrix)
    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0
    ########################
    self._envStepCounter = 0
    # Compute xyz point cloud from depth
    nx, ny = (self._width, self._height)
    x_index = np.linspace(0, nx - 1, nx)
    y_index = np.linspace(0, ny - 1, ny)
    self.xx, self.yy = np.meshgrid(x_index, y_index)
    self.xx -= float(nx) / 2
    self.yy -= float(ny) / 2
    self._camera_fx = self._width / 2.0 / np.tan(fov / 2.0 / 180.0 * np.pi)
    self._camera_fy = self._height / 2.0 / np.tan(fov / 2.0 / 180.0 * np.pi)
    self.xx /= self._camera_fx
    self.yy /= self._camera_fy
    self.xx *= -1.0

    # initial_q_list = [-0.3155566399904822, -1.4619890157831814, -0.12641376890399098, -3.000872652806791, -0.3591005122203335, 3.1289595169004025, 0.2756560279943101]
    # self.robot.setJointValue(initial_q_list,210)

    # self.start_orn = self.p.getJointState(self.obj_id,0)[0]
    # cur_pos = self.robot.getWrenchTipPos()
    # self.prev_dist = np.linalg.norm(target_pos - cur_pos)
    # self.prev_orn = np.copy(self.start_orn)
    # return self._get_observation()

  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
    return np_img_arr

  def take_pic(self, compress=False):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    if compress:
      np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
    return np_img_arr

  def _get_observation_img(self):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    return np_img_arr


  def _get_observation_imgseg(self):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
    seg = img_arr[4][:-10,20:]
    return np_img_arr, seg

  def angleaxis2quaternion(self,angleaxis):
    angle = np.linalg.norm(angleaxis)
    axis = angleaxis / (angle + 0.00001)
    q0 = cos(angle/2)
    qx,qy,qz = axis * sin(angle/2) 
    return np.array([qx,qy,qz,q0])

  def quaternion2angleaxis(self,quater):
    angle = 2 * acos(quater[3])
    axis = quater[:3]/(sin(angle/2)+0.00001)
    angleaxis = axis * angle
    return np.array(angleaxis)

  def step(self, action):
    next_pos = np.array(self.robot.getEndEffectorPos()) + np.array(action)[:3]
    next_cur = np.array(self.robot.getEndEffectorOrn())
    next_cur = np.array(self.p.getEulerFromQuaternion(self.robot.getEndEffectorOrn()))
    next_cur[0] += action[3]
    next_cur[1] += action[4]
    next_cur[2] += action[5]
    orn_next = self.p.getQuaternionFromEuler(next_cur)
    for _ in range(4):
        self.robot.operationSpacePositionControl(next_pos,orn=orn_next,null_pose=None,gripperPos=220)
    observation, seg = self._get_observation_imgseg()
    reward,done,suc = self._reward()
    return observation, reward, done, suc


  def _reward(self):
    self.termination_flag = False
    self.success_flag = False

    reward = 0
    cur_pos_L = self.robot.getWrenchLeftTipPos()    
    cur_pos_L[2] = self.p.getAABB(self.robot.robotId, self.robot.wrench_left_tip_index)[0][2] 

    target_pos = np.array(self.p.getLinkState(self.obj_id,0)[0])
    dist_L = np.linalg.norm(target_pos - cur_pos_L)

    cur_pos_R = self.robot.getWrenchRightTipPos()    
    cur_pos_R[2] = self.p.getAABB(self.robot.robotId, self.robot.wrench_right_tip_index)[0][2] 
    dist_R = np.linalg.norm(target_pos - cur_pos_R)
    
    dist = 0.5 * dist_L + 0.5 * dist_R
    cur_orn = self.p.getJointState(self.obj_id,0)[0]
    reward_orn = self.prev_orn - cur_orn
    self.prev_orn = cur_orn

    next_cur = np.array(self.p.getEulerFromQuaternion(self.robot.getWrenchLeftTipOrn()))
    
    reward = reward_orn * 100.0
    self._env_step += 1

    if self.start_orn - cur_orn > 30/180.0*math.pi:
      self.termination_flag = True
      self.success_flag = True
      reward = 5.0
    if dist > 0.04:
      self.termination_flag = True
      self.success_flag = False
      reward = -1.0
    if self._env_step >= self._maxSteps:
      self.termination_flag = True
      self._env_step = 0
    return reward, self.termination_flag, self.success_flag

  def get_camera_image(self, params_dict,width=640, height=480, target_position=[0, 0, 0], hook_world_pos=None, compress=False):
      fov = 60
      far = 4
      near = 0.02
      aspect = 1.
      cameraUpVector = np.array(params_dict['up_vector'], dtype=np.float64)
      cameraEyePosition = np.array(params_dict['eye_position'], dtype=np.float64)
      if not (hook_world_pos is None):
        cameraEyePosition += np.array(hook_world_pos)
      # cameraUpVector = np.array([1, 0.0, 0.0])
      # cameraEyePosition = np.array([table_center_x,table_center_y,2])


      nx, ny = (width, height)
      x_index = np.linspace(0, nx - 1, nx)
      y_index = np.linspace(0, ny - 1, ny)
      xx, yy = np.meshgrid(x_index, y_index)
      xx -= float(nx) / 2
      yy -= float(ny) / 2
      camera_fx = width / 2.0 / np.tan(fov / 2.0 / 180.0 * np.pi)
      camera_fy = height / 2.0 / np.tan(fov / 2.0 / 180.0 * np.pi)
      xx /= camera_fx
      yy /= camera_fy
      xx *= -1.0

      proj_matrix = np.array(self.p.computeProjectionMatrixFOV(fov, aspect, near, far))
      cameraTargetPosition = np.array(target_position)
      if not (hook_world_pos is None):
        cameraTargetPosition += np.array(hook_world_pos)

      view_matrix = self.p.computeViewMatrix(cameraEyePosition,cameraTargetPosition,cameraUpVector)
      view_matrix_np = np.array(view_matrix)
      view_matrix_np = view_matrix_np.reshape((4,4))
      view_matrix_inv = np.linalg.inv(view_matrix_np)
      
      
      img_arr = self.p.getCameraImage(width=width + 20,
                        height=height + 10,
                        viewMatrix=view_matrix,
                        projectionMatrix=proj_matrix)

      rgb = img_arr[2][:-10,20:,:3]
      np_img_arr = np.reshape(rgb, (height, width, 3))
      if compress:
        np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
      return np_img_arr
