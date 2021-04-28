import pybullet as p

def debug_sphere(loc, radius=0.2):
    p.createVisualShape(p.GEOM_SPHERE,radius=0.2, specularColor=[0.4, .4, 0])
