import sys
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import bullet_client as bc
from coord_helper import * 
from data_helper import * 
from collision_helper import fcl_get_dist, fcl_model_to_fcl

import imageio
def check_one_pose_simple(p, hook_world_pos, hook_bullet_id, object_bullet_id, ori_transl, ori_quat, hook_urdf, object_urdf, fcl_hook_model, fcl_object_model):
	failure = False
	ori_object_pos = ori_transl

	if (not(fcl_hook_model is None)) and (not(fcl_object_model) is None):
		fcl_model_h = fcl_model_to_fcl(fcl_hook_model['obj_v'], fcl_hook_model['obj_f'])
		fcl_model_o = fcl_model_to_fcl(fcl_object_model['obj_v'], fcl_object_model['obj_f'])
		fcl_dist = fcl_get_dist(fcl_model_h, fcl_model_o, ori_transl, ori_quat)
		if fcl_dist == 0:
			return False, np.append(ori_transl, ori_quat)

	for i in range(100):
		#check overlap of model with hook
		# if i == 0:
		# 	for tmp in p.getClosestPoints(hook_bullet_id, object_bullet_id, 0.003):
		# 		if tmp[8] < -0.001:
		# 			failure = True
		# 			print('penetration problem')
		# 			break
		# 	if failure:
		# 		break
		p.stepSimulation()		

		if ((i+1)%1 == 0 and i < 10) or (i%5 == 0 and i >= 10):
			object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
			object_pos = object_pos_world - hook_world_pos

			#check overlap of model with hook
			# for tmp in p.getClosestPoints(hook_bullet_id, object_bullet_id, 0.003):
			# 	if tmp[8] < -0.001:
			# 		failure = True
			# 		break
			# if failure:
			# 	break
			
			# if object center too low or object too far away
			if object_pos_world[2] < 0.2 or np.linalg.norm(object_pos_world) > 5:
				# print('pos problem')
				failure = True
				break
			
			# if touches ground
			if check_object_touches_ground(object_bullet_id, p):
				failure = True
				break

			# too much change in pos
			if ori_object_pos[2] - object_pos[2] > 0.6:
				failure = True
				break
		
	object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
	object_pos = object_pos_world - hook_world_pos
	return (not failure), np.append(object_pos, np.array(object_quat))
