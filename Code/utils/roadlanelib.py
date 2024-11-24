
# from mathutils import Matrix , Vector
import numpy as np
from scipy.interpolate import splprep, splev
from utils.lib import find_xyz
import bpy

def func_filter_3d_points(lane_3d_pts):
    filtered_lane_3d_pts = []
    for i in range(1, len(lane_3d_pts)):
        if lane_3d_pts[i][2] - lane_3d_pts[i - 1][2] < 0:
            break
        else:
            filtered_lane_3d_pts.append(lane_3d_pts[i])
    return filtered_lane_3d_pts

def sub_sample_points(points, n=5):
    sub_sampled_points = []
    # Sub sample the points
    n = min(n, len(points)//2)
    for i in range(0, len(points), n):
        sub_sampled_points.append(points[i])
    return sub_sampled_points

def get_3d_lane_pts(R, K, lane_points, depth, lane_bbox, scale_factor):
    lane_points = np.array(lane_points)
    lane_bbox = np.array(lane_bbox)
    lane_3d_pts = []
    lane_3d_bbox = []
    # Take first 75 percent only 
    for i in range(lane_points.shape[0]):
        point = lane_points[i]
        x = point[0]
        y = point[1]
        if x>=depth.shape[1]:
            x = depth.shape[1] - 1
        if y>=depth.shape[0]:
            y = depth.shape[0] - 1
        z = depth[int(y), int(x)]
        (x, y, z) = find_xyz(R, K, (x, y), z)
        x = x*scale_factor
        y = y*scale_factor
        z = z*scale_factor
        #fit_bezier_curvex = x*4 + 5
        # z = z
        lane_3d_pts.append((x, y, z))
        
    for i in range(len(lane_bbox)):
        point = lane_bbox[i]
        x = point[0]
        y = point[1]
        if x>=depth.shape[1]:
            x = depth.shape[1] - 1
        if y>=depth.shape[0]:
            y = depth.shape[0] - 1
        z = depth[int(y), int(x)]
        (x, y, z) = find_xyz(R, K, (x, y), z)
        x = x*scale_factor
        y = y*scale_factor
        z = z*scale_factor
        lane_3d_bbox.append((x, y, z))
    
    
    return lane_3d_pts, lane_3d_bbox
