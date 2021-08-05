import os
import json
import shutil

partnet_symph_dir = '/home/yifany/partnet_symph'
category = 'Scissors'
labels_dir = '/home/yifany/geo-hook/scripts/notes/partnet_labels_scissors.txt'

unzip_dir = os.path.join(partnet_symph_dir, category, 'models')

id_dict = {}
obj_id = 1
for obj_name in sorted(os.listdir(unzip_dir)):
print(obj_name)
    obj_id_partnet = int(obj_name[:-4])
    obj_dir = os.path.join(unzip_dir, obj_name)
    
    obj_folder_dir = os.path.join(unzip_dir, str(obj_id))
    os.mkdir(obj_folder_dir)
    obj_dir_new = os.path.join(obj_folder_dir, 'model.obj')

    shutil.copyfile(obj_dir, obj_dir_new)
    id_dict[obj_id] = obj_id_partnet
    obj_id += 1

with open(labels_dir, 'a+') as f:
    f.write(json.dumps(id_dict))


    