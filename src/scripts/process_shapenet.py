
import os
import pybullet as p
from normalize_obj import normalize_one_obj
import sys
import subprocess
import json
from distutils.dir_util import copy_tree

ori_shapenet_dir = '/juno/group/linshao/ShapeNetCore'
shapenet_dir = '/scr1/yifan/shapenet_partial'
shapenet_new_dir = '/scr1/yifan/geo_data'
category_dict = {
    'bag': '02773838',
    'cap': '02954340',
    'headphone': '03261776',
    'knife': '03624134',
    'mug': '03797390',
}
all_labels_dir = '/scr1/yifan/geo-hook/scripts/notes'

def process_one_category(categoy):
    unzip_dir = os.path.join(shapenet_dir, category)
    unzip_new_dir = os.path.join(shapenet_new_dir, category)
    labels_dir = os.path.join(all_labels_dir, 'shapenet_labels_{}.txt'.format(category))
    obj_id = 1
    id_dict = {}
    for obj_name in sorted(os.listdir(unzip_dir)):
        obj_dir = os.path.join(unzip_dir, obj_name, 'model.obj')
        obj_folder_dir = os.path.join(unzip_dir, obj_name)
        obj_normalized_dir = os.path.join(unzip_dir, obj_name, 'model_normalized.obj')
        obj_v_dir = os.path.join(unzip_dir, obj_name, 'model_normalized_v.obj')
        if os.path.exists(obj_dir):
            if not os.path.exists(obj_normalized_dir):
                try:
                    normalize_one_obj(obj_dir, obj_normalized_dir)
                except Exception as e:
                    continue
            
            if not os.path.exists(obj_v_dir):
                print('converting')
                os.system('python to_vhacd.py {}'.format(obj_normalized_dir))
                if os.path.exists(obj_v_dir):
                    print('success')
                # p.vhacd(obj_normalized_dir, obj_v_dir,
                    # os.path.join(unzip_dir, obj_folder, 'vhacd_log.txt'))

            obj_id_shapenet = obj_name[:-4]
    
            obj_folder_new_dir = os.path.join(unzip_new_dir, str(obj_id))
            os.makedirs(obj_folder_new_dir)
            copy_tree(obj_folder_dir, obj_folder_new_dir)
            print(obj_folder_new_dir)
    
            id_dict[obj_id] = obj_id_shapenet
            obj_id += 1

    with open(labels_dir, 'w+') as f:
        f.write(json.dumps(id_dict))

                    
for category in category_dict.keys():
    process_one_category(category)