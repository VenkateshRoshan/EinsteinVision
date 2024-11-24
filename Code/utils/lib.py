
import numpy as np

def find_xyz(R, K, pts, depth):
    
    u ,v = pts
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    xyz = np.array([x, y, z , 1]).T
    xyz = np.dot(R, xyz)
    xyz = xyz[:3]

    return xyz

