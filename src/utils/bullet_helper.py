import pybullet
import sys
import os
import bullet_client as bc

from data_helper import * 
from coord_helper import *
# from train_helper import *
from rotation_lib import *

try:
	import cv2
except:
	pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p_init_multithread(bi, gui):
	if gui and bi == 0:
		p_tmp = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p_tmp = bc.BulletClient(connection_mode=pybullet.DIRECT)
	p_enable_physics(p_tmp)
	return p_tmp

def p_reset_multithread(bi, p_list, gui):
	if p_list[bi].isConnected():
		p_list[bi].disconnect()
	if gui and bi == 0:
		p_tmp = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p_tmp = bc.BulletClient(connection_mode=pybullet.DIRECT)
	p_enable_physics(p_tmp)
	
	return p_tmp
def p_enable_physics(p):
	p.setPhysicsEngineParameter(enableConeFriction=1)
	p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
	p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
	p.setPhysicsEngineParameter(numSolverIterations=40)
	p.setPhysicsEngineParameter(numSubSteps=40)
	p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
	p.setPhysicsEngineParameter(enableFileCaching=0)
	p.setTimeStep(1 / 100.0)
	p.setGravity(0,0,-9.81)
def p_draw_ball(p, center, radius=0.03):
	ball_c = p.createCollisionShape(p.GEOM_SPHERE,radius=radius)
	ball_v = p.createVisualShape(p.GEOM_SPHERE, radius=radius)
	ball_id = p.createMultiBody(0, ball_c, ball_v, basePosition=center) 
	return ball_id

def p_reset(p, gui=False, physics=False):
	if not (p is None):
		p.disconnect()
	if gui:
		p = bc.BulletClient(connection_mode=pybullet.GUI)
	else:
		p = bc.BulletClient(connection_mode=pybullet.DIRECT)

	if physics:
		p_enable_physics(p)
	
	return p

def p_carpet(p):
	resource_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'resource'))
	texture_dir = os.path.join(resource_dir,"texture")
	urdf_dir = os.path.join(resource_dir,"urdf")

	plane_id = p.loadURDF(os.path.join(urdf_dir,"plane.urdf"),[0,0,0])
	env_textid = p.loadTexture(os.path.join(texture_dir,"texture1.jpg"))
	p.changeVisualShape(0, -1, textureUniqueId=env_textid)

def p_photo(p, params_dict=None, default=True, width=640, height=640):
	assert (default == False) ^ (params_dict is None)
	if default:
		# params_dict = {
		# 	'up_vector':  [0.3, 0, 0.05],
		# 	'eye_position': np.array([-0.3, 0, 0.3]) + np.array([0.7, 0, 1]),
		# 	'target_pos': np.array([0.7, 0, 1])
		# }
		params_dict = {
			'up_vector':  [0.3, 0.2, 0.1],
			'eye_position': np.array([-0.2, -0.2, 0.2]) + np.array([0.7, 0, 1]),
			'target_pos': np.array([0.7, 0, 1])
		}

	fov = 60
	far = 4
	near = 0.02
	aspect = 1.
	cameraUpVector = np.array(params_dict['up_vector'], dtype=np.float64)
	cameraEyePosition = np.array(params_dict['eye_position'], dtype=np.float64)
	if not 'target_pos' in params_dict:
		cameraTargetPosition = np.array([0, 0, 0])
	else:
		cameraTargetPosition = np.array(params_dict['target_pos'], dtype=np.float64)
		

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

	proj_matrix = np.array(p.computeProjectionMatrixFOV(fov, aspect, near, far))
	view_matrix = p.computeViewMatrix(cameraEyePosition,cameraTargetPosition,cameraUpVector)
	view_matrix_np = np.array(view_matrix)
	view_matrix_np = view_matrix_np.reshape((4,4))
	view_matrix_inv = np.linalg.inv(view_matrix_np)
	
	img_arr = p.getCameraImage(width=width + 20,
									  height=height + 10,
									  viewMatrix=view_matrix,
									  projectionMatrix=proj_matrix,
									  renderer=p.ER_TINY_RENDERER)
	rgb = img_arr[2][:-10,20:,:3]
	np_img_arr = np.reshape(rgb, (height, width, 3))

	return np_img_arr

def p_partial_pc(p, obj_id, params_dict,width=640, height=480):
	fov = 60
	far = 4
	near = 0.02
	aspect = 1.
	cameraUpVector = np.array(params_dict['up_vector'], dtype=np.float64)
	cameraEyePosition = np.array(params_dict['eye_position'], dtype=np.float64)
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


	proj_matrix = np.array(p.computeProjectionMatrixFOV(fov, aspect, near, far))
	cameraTargetPosition = np.array([0, 0, 0])
	view_matrix = p.computeViewMatrix(cameraEyePosition,cameraTargetPosition,cameraUpVector)
	view_matrix_np = np.array(view_matrix)
	view_matrix_np = view_matrix_np.reshape((4,4))
	view_matrix_inv = np.linalg.inv(view_matrix_np)
	
	
	img_arr = p.getCameraImage(width=width,
									  height=height,
									  viewMatrix=view_matrix,
									  projectionMatrix=proj_matrix)
	rgb = img_arr[2]
	np_img_arr = np.reshape(rgb, (height, width, 4))
	image_rgb = np_img_arr[:, :, :3]
	depth = (img_arr[3] - 0.5) * 2.0
	depth = 2 * far * near /( depth * (far - near) - (far + near))
	# depth = far * near / (far - (far - near) *img_arr[3])
	# print('depth shape', depth.shape)
	seg = img_arr[4]
	seg = np.expand_dims(seg,axis=-1)
	cam_z = depth
	cam_x = xx * cam_z
	cam_y = yy * cam_z
	image_xyz = np.dstack([cam_x,cam_y,cam_z,np.ones_like(cam_x)])
	image_xyz = np.matmul(image_xyz,view_matrix_inv)[:,:,0:3]
	# image_xyz = np.matmul(image_xyz,_view_matrix_inv)[:,:,0:3]
	obj_labeling = np.zeros((height, width,4))
	id_labeling = obj_labeling[:,:,3:4]
	obj_center = obj_labeling[:,:,0:3]
	id_labeling[seg == obj_id] = 1

	img_xyz_pc = image_xyz.reshape((-1,3))
	img_rgb_pc = image_rgb.reshape((-1,3))
	img_seg_pc = id_labeling.reshape((-1,1))
	pc_index = np.where(np.all(img_seg_pc,axis=1))[0]
	xyz_id = img_xyz_pc[pc_index,:]
	return xyz_id

class p_Env:
	def __init__(self, home_dir_data, gui=False, physics=False):
		self.gui = gui
		self.physics = physics
		self.p = None
		self.p = p_reset(self.p, gui=self.gui, physics=self.physics)
		p_carpet(self.p)

		self.home_dir_data = home_dir_data
		self.data_dir = os.path.join(self.home_dir_data, 'geo_data')
		hook_dict, object_dict = load_all_hooks_objects(self.data_dir, ret_dict=True, print_num=False)
		self.hook_dict = hook_dict
		self.object_dict = object_dict
			
		self.cur_hook_name = None
		self.cur_object_name = None
		self.hook_bullet_id = None
		self.object_bullet_id = None
		self.hook_world_pos = np.array([0.7, 0, 1])

		self.refresh_ct = 0


	def load_pair(self, hook_name, object_name):
		if self.refresh_ct % 5 == 0 and self.refresh_ct != 0:
			self.p = p_reset(self.p, gui=self.gui, physics=self.physics)
			p_carpet(self.p)
			self.cur_hook_name = None
			self.cur_object_name = None
			self.hook_bullet_id = None
			self.object_bullet_id = None

		if not (hook_name == self.cur_hook_name and object_name == self.cur_object_name):

			if not (self.hook_bullet_id is None):
				self.p.removeBody(self.hook_bullet_id)
				self.p.removeBody(self.object_bullet_id)

			self.hook_bullet_id = self.p.loadURDF(self.hook_dict[hook_name]['urdf'], basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
			self.object_bullet_id = self.p.loadURDF(self.object_dict[object_name]['urdf'], basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=False)
				
			self.cur_hook_name = hook_name
			self.cur_object_name = object_name

			self.refresh_ct += 1
	def set_object_pose(self, pose_transl, pose_quat, aa=False):
		if aa:
			pose_quat = quaternion_from_angle_axis(pose_quat)
		self.p.resetBasePositionAndOrientation(self.object_bullet_id, pose_transl + self.hook_world_pos, pose_quat)        

	def load_pair_w_pose(self, result_file_name, pose_transl, pose_quat, aa=False):
		hook_name, object_name = split_result_file_name(result_file_name)
		self.load_pair(hook_name, object_name)        
		self.set_object_pose(pose_transl, pose_quat, aa=aa)

	def photo(self, save_dir=None, params_dict=None, default=True, width=640, height=640):
		img = p_photo(self.p, params_dict=None, default=True, width=width, height=height)
		if not (save_dir is None):
			cv2.imwrite(save_dir, img)
		else:
			return img
	
