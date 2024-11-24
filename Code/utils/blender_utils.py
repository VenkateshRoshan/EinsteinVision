import bpy 
import math 
import json 
import numpy as np 
import pickle
from mathutils import Matrix, Vector 

def delete_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_camera():
    # Create a new camera if none exists
    if not bpy.data.objects.get('Camera'):
        bpy.ops.object.camera_add()
    # Get the camera object
    camera = bpy.data.objects.get('Camera')  # Assuming the camera's name is 'Camera'
    return camera

def transform_camera_world(pts, R, K):
    # Return World Coordinates of the points
    # R is a 4*4 matrix 
    # K is a 3*3 matrix
    # pts is a tuple of (u, v)
    
    u, v = pts
    xyz = np.dot(np.array([u, v, 1]), np.linalg.inv(K).dot(np.linalg.inv(R)[:3, :4]))
    
def get_hardcoded_KR():
    K = np.array([[1622.30674706393, 0.0, 681.0156669556608],
                  [0.0, 1632.8929856491513, 437.0195537829288],
                  [0.0, 0.0, 1.0]])
    
    R = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1.5],
                    [0, 0, 0, 1]])
    
    return K, R

def open_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
    