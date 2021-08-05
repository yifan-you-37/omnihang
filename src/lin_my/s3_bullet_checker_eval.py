import sys
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import bullet_client as bc
from coord_helper import * 
from data_helper import * 
import time

import imageio

def check_one_pose_simple(p, hook_world_pos, hook_bullet_id, object_bullet_id, ori_transl, ori_quat, hook_urdf, object_urdf, fcl_hook_model, fcl_object_model):
	failure = False
	ori_object_pos = ori_transl
	non_contact_count = 0

	for i in range(100):
		if i == 0:
			start_time = time.time()
		else:
			ssecond = time.time() - start_time
			# if ssecond > 60:
				# failure = True
				# break

		#check overlap of model with hook
		# if i == 0:
		# 	L1 = p.getContactPoints(hook_bullet_id, object_bullet_id, linkIndexA=0, linkIndexB=-1)
		# 	# print("Length of zero hook",len(L1))
		# 	if len(L1) > 0:
		# 		for tmp in L1:
		# 			if tmp[8] < -0.003:
		# 				failure = True
		# 				print('penetration problem')
		# 				break
		# 	# print("finished geting closesetpoints")
		# 	if failure:
		# 		break
		
		p.stepSimulation()		
		if 1:
			hook_AABB = p.getAABB(hook_bullet_id,0)
			object_AABB = p.getAABB(object_bullet_id)
			if (hook_AABB[0][0] > object_AABB[1][0] or hook_AABB[1][0] < object_AABB[0][0]) \
				or (hook_AABB[0][1] > object_AABB[1][1] or hook_AABB[1][1] < object_AABB[0][1]) \
				or (hook_AABB[0][2] > object_AABB[1][2] or hook_AABB[1][2] < object_AABB[0][2]): 
				failure = True
				break
		if ((i+1)%1 == 0 and i < 10) or (i%5 == 0 and i >= 10):
			object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
			object_pos = object_pos_world - hook_world_pos
			CP_List = p.getContactPoints(bodyA=hook_bullet_id, bodyB=object_bullet_id, linkIndexA=0,linkIndexB=-1)	
			if len(CP_List) > 0:
				for tmp in CP_List:
					if tmp[8] < -0.003:
						failure = True
						break
			if failure:
				break
		
			# if object center too low or object too far away
			if object_pos_world[2] < 0.2 or np.linalg.norm(object_pos_world) > 5:
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
	
	ssecond = time.time() - start_time

	object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
	object_pos = object_pos_world - hook_world_pos
	return (not failure), np.append(object_pos, np.array(object_quat))
	

# def check_one_pose_simple(p, hook_world_pos, hook_bullet_id, object_bullet_id, ori_transl, ori_quat, hook_urdf, object_urdf, fcl_hook_model, fcl_object_model):
# 	failure = False
# 	ori_object_pos = ori_transl
# 	non_contact_count = 0

# 	thres = 0.01
# 	ratio = 1	
# 	for i in range(100):
# 		thres = (thres - 0.01) * (100 - i) / 100.0 + 0.01
# 		#print("step in simualtion",i,"thres",thres)
# 		if i == 0:
# 			#time.sleep(3)
# 			start_time = time.time()
# 		else:
# 			ssecond = time.time() - start_time
# 			#print("time spend",ssecond)
# 			if ssecond > 60:
# 				failure = True
# 				break
# 		#check overlap of model with hook
# 		if i == 0:
# 			#L1 = p.getContactPoints(hook_bullet_id, object_bullet_id, linkIndexA=-1, linkIndexB=-1)
# 			L1 = p.getContactPoints(hook_bullet_id, object_bullet_id, linkIndexA=0, linkIndexB=-1)
# 			# print("Length of zero hook",len(L1))
# 			if len(L1) > 0:
# 				for tmp in L1:
# 					if len(tmp) >= 8 and tmp[8] < -thres * ratio:
# 						failure = True
# 						print('penetration problem')
# 						break
# 			# print("finished geting closesetpoints")
# 			if failure:
# 				break
# 		p.stepSimulation()		
# 		#p.changeDynamics(object_bullet_id, -1, contactStiffness=1.0+i, contactDamping=0.01)
# 		# print(p, p._client, hook_bullet_id, object_bullet_id)
# 		if 1:
# 			hook_AABB = p.getAABB(hook_bullet_id,0)
# 			object_AABB = p.getAABB(object_bullet_id)
# 			if (hook_AABB[0][0] > object_AABB[1][0] or hook_AABB[1][0] < object_AABB[0][0]) \
# 				or (hook_AABB[0][1] > object_AABB[1][1] or hook_AABB[1][1] < object_AABB[0][1]) \
# 				or (hook_AABB[0][2] > object_AABB[1][2] or hook_AABB[1][2] < object_AABB[0][2]): 
# 				failure = True
# 				break
# 		if ((i+1)%1 == 0 and i < 10) or (i%5 == 0 and i >= 10):
# 			object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
# 			object_pos = object_pos_world - hook_world_pos
# 			CP_List = p.getContactPoints(bodyA=hook_bullet_id, bodyB=object_bullet_id, linkIndexA=0,linkIndexB=-1)	
# 			# print("getHookClosest len",len(CP_List))	
# 			if len(CP_List) > 1:
# 				cp_distance = [tmp[8] for tmp in CP_List]
# 		#		print("cp_distance",np.min(np.array(cp_distance)),np.array(cp_distance).shape)
# 	#			print(np.array(cp_distance))
# 				failure = True
# 				if len(cp_distance) < 8 or i < 50:
# 					failure = False
# 				else:
# 					for c in cp_distance:
# 						if c > 0:
# 							failure = False
# 							break
# 				if failure:
# 	#				print("all negative!!!!")
# 					break
# 			count_cp = 0
# 			if len(CP_List) > 0:
# 				for tmp in CP_List:
# 					if len(tmp) >= 8 and tmp[8] < 0.0:
# 						count_cp += 1.0
# 					if tmp[8] < -thres * ratio:
# 						failure = True
# 					#	print('penetration problem')
# 						break
# 			if failure:
# 				break
		
# 			# if object center too low or object too far away
# 			if object_pos_world[2] < 0.2 or np.linalg.norm(object_pos_world) > 5:
# 				#print('pos problem')
# 				failure = True
# 				break
			
# 			# if touches ground
# 			if check_object_touches_ground(object_bullet_id, p):
# 				#print('touches problem')
# 				failure = True
# 				break

# 			# too much change in pos
# 			if ori_object_pos[2] - object_pos[2] > 0.6:
# 				#print('too much change problem')
# 				failure = True
# 				break
# 		# too much change in quat
# 		# print(i)
	
# 	ssecond = time.time() - start_time
# 	# print("time spend",ssecond)

# 	object_pos_world, object_quat = p.getBasePositionAndOrientation(object_bullet_id)
# 	object_pos = object_pos_world - hook_world_pos
# 	return (not failure), np.append(object_pos, np.array(object_quat))
	

