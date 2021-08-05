import numpy as np


def obj_loader(file_name,normalize=False):
  """It loads the vertices, vertice normals, the faces of a wavefront obj file.
  """
  vertices = []
  faces = []
  vnormals = []

  with open(file_name,'r') as fin:
    for line in fin:
      if line.startswith('#'):
        continue
      values = line.split()
      if len(values) < 1:
        continue
      if values[0] == 'v':
        v = list(map(float,values[1:4]))
        vertices.append(v)
      elif values[0] == 'vn':
        vn = list(map(float,values[1:4]))
        vnormals.append(vn)
      elif values[0] == 'f':
        face = []
        for v in values[1:]:
          w = v.split('/')
          face.append(int(w[0])) 
        faces.append(face) 
  vertices = np.array(vertices)
  faces = np.array(faces)
  vnormals = np.array(vnormals)
  faces = faces-1
  if normalize:
    bbox_max = np.max(vertices,axis=0)
    bbox_min = np.min(vertices,axis=0)
    bbox_center = 0.5 * (bbox_max + bbox_min)
    bbox_rad =  np.linalg.norm(bbox_max - bbox_center)
    vertices -= bbox_center
    vertices /= (bbox_rad*2.0)
  if np.any(faces < 0):     
    print('Negative face indexing in obj file')
  return vertices, faces, vnormals

def obj_exporter(vertices, faces, out_dir, normalize_pos=False):
  # if normalize_pos:
  #   bbox_max = np.max(vertices,axis=0)
  #   bbox_min = np.min(vertices,axis=0)
  #   bbox_center = 0.5 * (bbox_max + bbox_min)
  #   vertices -= bbox_center
    # print(bbox_center)
  faces += 1
  with open(out_dir, 'w+') as f:
    for i in range(vertices.shape[0]):
      f.write('v {:.8f} {:.8f} {:.8f}\n'.format(vertices[i][0], vertices[i][1], vertices[i][2]))
    for i in range(faces.shape[0]):
      face = faces[i]
      f.write('f {} {} {}\n'.format(int(face[0]), int(face[1]), int(face[2])))
  # if normalize_pos:
    # return bbox_center

def merge_obj(v1, f1, v2, f2):
  v = np.concatenate((v1, v2), axis=0)
  f2 += v1.shape[0]
  f = np.concatenate((f1, f2), axis=0)
  return v, f

def get_bb(file_name):
  vertices = []

  with open(file_name,'r') as fin:
    for line in fin:
      if line.startswith('#'):
        continue
      values = line.split()
      if len(values) < 1:
        continue
      if values[0] == 'v':
        v = list(map(float,values[1:4]))
        vertices.append(v)
  vertices = np.array(vertices)

  bbox_max = np.max(vertices,axis=0)
  bbox_min = np.min(vertices,axis=0)
  bbox_center = 0.5 * (bbox_max + bbox_min)
  # bbox_rad =  np.linalg.norm(bbox_max - bbox_center)

  return bbox_center, bbox_max - bbox_center

if __name__ == "__main__":
  v,f,vn = obj_loader('/Users/lins/GraspNet/mesh_processing/reorient_faces_coherently/vase.obj',normalize=True)
