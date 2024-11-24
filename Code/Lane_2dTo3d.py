
import numpy as np
import joblib
import pickle
from utils.roadlanelib import sub_sample_points, get_3d_lane_pts, func_filter_3d_points
from DepthFinder import getDepthFromImage
import torch
from tqdm import tqdm

zoe = torch.hub.load("../ZoeDepth/", "ZoeD_NK", source="local", pretrained=True)
# zoe = zoe.to("cuda")
K = np.array([[1622.30674706393,0.0,681.0156669556608],
             [0.0,1632.8929856491513,437.0195537829288],
             [0.0,0.0,1.0]])

R = np.array([[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1.5],
    [0, 0, 0, 1]])

def conv_2dTo3d(path,img_path) :
    
    with open(path, 'rb') as f:
        lane_data = pickle.load(f)

    Lane_3d_pts = {}

    for l_key in tqdm(lane_data.keys()):
        lane = lane_data[l_key]['final_lanes']
        LANE_3D_PTS = []
        LANE_CLASSES = []
        frame_name = l_key.split('/')[-1]
        for i in lane:
            single_lane = i
            lane_points = single_lane[0]
            lane_bbox = single_lane[1]
            lane_class = single_lane[2]

            lane_subsampled_points = sub_sample_points(lane_points, 1)
            
            dep = getDepthFromImage(zoe,img_path+frame_name)
            depth_scale_factor = (max(dep.ravel()) - min(dep.ravel())) / max(dep.ravel())

            lane_3d_pts, lane_3d_bbox = get_3d_lane_pts(R, K, lane_subsampled_points, dep, lane_bbox, depth_scale_factor)
            sorted(lane_3d_pts, key=lambda x: x[2])
            lane_3d_pts = func_filter_3d_points(lane_3d_pts)
            LANE_3D_PTS.append(lane_3d_pts)
            LANE_CLASSES.append(lane_class)
        Lane_3d_pts[frame_name] = {}
        Lane_3d_pts[frame_name]['final_lanes'] = LANE_3D_PTS
        Lane_3d_pts[frame_name]['final_lane_classes'] = LANE_CLASSES

    return Lane_3d_pts
    joblib.dump(Lane_3d_pts, f'Images_')

def main():
    Lane_Path = 'Data/Lanes/'
    IMAGE_PATH = 'Data/FinalImages/'
    
    for i in tqdm(range(7,14)) :
        Lane_3d_pts = conv_2dTo3d(Lane_Path + f'results_video{i}.pkl',IMAGE_PATH+f'Images_{i}/')
        joblib.dump(Lane_3d_pts, f'Images_{i}_3d.pkl')

if __name__ == '__main__':
    main()