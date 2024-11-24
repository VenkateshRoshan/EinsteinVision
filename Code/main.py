import bpy
import os
from mathutils import Matrix , Vector
import numpy as np
import json
import pickle
from scipy.interpolate import splprep, splev
import joblib
import argparse
import sys
from tqdm import tqdm
import cv2
import torch
sys.path.append('../ZoeDepth/')
from zoedepth.utils.misc import get_image_from_url, colorize
import easyocr

sys.path.append('..')

from BlenderResources import BlenderSceneGenerator, addTexture
from utils.lib import find_xyz
from utils.roadlanelib import *
from DepthFinder import getDepthFromImage # To Find Depth Values
from raftInference import * # To find Optical Flow
from ocrInference import run as findText # To find Text in Image

YOLOCLASSES_TO_BLENDER = {
    'car': 'Car' ,
    # 'motorcycle': 'B_Wheel',
    'bicycle': 'roadbike 2.0.1',
    'dustbin': 'Bin_Mesh.072',
    'trash can': 'Bin_Mesh.072',
    'stop sign': 'StopSign_Geo',
    'parking meter': 'sign_25mph_sign_25mph',
    'yellow traffic light': 'TrafficSignalYellow',
    'green traffic light': 'TrafficSIgnalGreen',
    'red traffic light': 'TrafficSignalRed',
    'person': 'BaseMesh_Man_Simple',
    'fire': 'fire hydrant',
    'traffic cone': 'absperrhut',
    'traffic cylinder': 'trafficCylinder',
    'suv':'Jeep_3_',
    'truck':'Truck',
    'hump':'SpeedHump',
    'carRed':'Car_Red',
    'truckRed':'Truck_Red',
    'pickupTruckRed':'PickupTruck_Red',
    'suvRed':'Jeep_3Red',
    'pickuptruck':'PickupTruck'
}

Blender_rot_scale = {
    'bicycle' : {'orientation': (1.57,0,0) , 'scale' : (.15,.15,.15)}, # Done
    'car' : {'orientation': (0,0,3.14) , 'scale' : (.015,.015,.015)}, # Done
    'dustbin' : {'orientation': (1.57,0,0) , 'scale' : (1,1,1)}, # Bin_Mesh.072
    'trash can' : {'orientation': (1.57,0,0) , 'scale' : (1.1,1.1,1.1)}, # Bin_Mesh.072
    'fire' : {'orientation': (0,0,0) , 'scale' : (.2,.2,.2)}, # Done
    'motorcycle' : {'orientation': (1.57,0,1.57) , 'scale' : (.006,.006,.006)}, # Done
    'speed limit sign' : {'orientation': (0,0,0) , 'scale' : (.6,.6,.6)}, # Done
    'parking meter' : {'orientation': (0,0,0) , 'scale' : (.6,.6,.6)}, # Done
    'person' : {'orientation': (1.57,0,3.14) , 'scale' : (.015,.015,.015)}, # Done
    'stop sign' : {'orientation': (4.71,3.14,1.57) , 'scale' : (.45,.45,.45)}, # Done
    'traffic cone' : {'orientation': (0,0,0) , 'scale' : (.2,.2,.2)}, # Done
    'traffic cylinder' : {'orientation': (0,0,0) , 'scale' : (.2,.2,.2)}, # Done
    'green traffic light' : {'orientation': (1.57,0,-1.57) , 'scale' : (.15,.15,.15)}, # Done
    'red traffic light' : {'orientation': (1.57,0,-1.57) , 'scale' : (.15,.15,.15)},   # Done
    'yellow traffic light' : {'orientation': (1.57,0,-1.57) , 'scale' : (.15,.15,.15)}, # Done
    'suv' : {'orientation': (0,0,0) , 'scale' : (2.5,2.5,2.5)}, # Done
    'truck' : {'orientation': (0,0,0) , 'scale' : (.0006,.0006,.0006)}, # Done
    'pickup truck' : {'orientation': (1.57,0,-1.57) , 'scale' : (.3,.3,.3)}, # Done
    'hump' : {'orientation': (0,1.57,0) , 'scale' : (.2,.2,2)}, # Done
    'carRed' : {'orientation': (0,0,3.14) , 'scale' : (.015,.015,.015)}, # Done
    'truckRed' : {'orientation': (0,0,0) , 'scale' : (.0006,.0006,.0006)}, # Done
    'pickupTruckRed' : {'orientation': (1.57,0,-1.57) , 'scale' : (.3,.3,.3)}, # Done
    'suvRed' : {'orientation': (0,0,0) , 'scale' : (2.5,2.5,2.5)}, # Done
}

# Objects Initializations

############################################################################################
K = np.array([[1622.30674706393,0.0,681.0156669556608],
             [0.0,1632.8929856491513,437.0195537829288],
             [0.0,0.0,1.0]])

R = np.array([[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1.5],
    [0, 0, 0, 1]])

zoe = torch.hub.load("../ZoeDepth/", "ZoeD_NK", source="local", pretrained=True)
reader = easyocr.Reader(['en'], gpu=True)

vehicle_types = ['car', 'truck', 'motorcycle', 'bicycle', 'suv']
traffic_light_types = ['red traffic light', 'green traffic light', 'yellow traffic light']
#############################################################################################

# Functions

def get_pose_details(frame_no, out_data):
    pose_details = []
    pose_bbox = []
    for i in list(out_data.keys()):
        if frame_no in out_data[i]['frame_ids']:
            frame_ids_list = list(out_data[i]['frame_ids'])
            frame_index = frame_ids_list.index(frame_no)
            pose_bbox.append(out_data[i]['bboxes'][frame_index])
            file_obj_path = "meshes/{0}/{1}.obj".format(str(i).zfill(4), str(frame_no).zfill(6))
            pose_details.append(file_obj_path)
    return pose_details, pose_bbox

def create_bezier_curve_from_points(points, name="Bezier_Curve"):
    # Create curve object in Blender without using Poly
    curve_data = bpy.data.curves.new(name="Bezier_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('BEZIER')
    polyline.bezier_points.add(len(points)-1)
    for i, (x, y, z) in enumerate(points):
        polyline.bezier_points[i].co = (x, z, 0)
        polyline.bezier_points[i].handle_left = (x, z, 0)
        polyline.bezier_points[i].handle_right = (x, z, 0)
    
    curve_object = bpy.data.objects.new(name="Bezier_Curve_Object", object_data=curve_data)
    bpy.context.collection.objects.link(curve_object)
    return curve_object

def create_lane_markings_by_curve_length(curve_object, lane_width=4, lane_length=10, gap_length=1, num_lanes=10):
    bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.scale[1] = 0.1 * lane_width
    bpy.context.object.scale[0] = 0.1* lane_length
    bpy.context.object.scale[2] = 0.05
    
    # Add modifier 
    bpy.ops.object.modifier_add(type='ARRAY')
    # Length of the lane markings array should be the length of the curve object
    bpy.context.object.modifiers["Array"].fit_type = 'FIT_CURVE'
    bpy.context.object.modifiers["Array"].curve = curve_object
    
    # Position lane markings along the curve object
    bpy.context.object.modifiers["Array"].use_constant_offset = True
    bpy.context.object.modifiers["Array"].constant_offset_displace[0] = gap_length
    bpy.ops.object.modifier_add(type='CURVE')
    bpy.context.object.modifiers["Curve"].object = curve_object
    
    # Currently deforming along Position X as it is the current fit 
    bpy.context.object.modifiers["Curve"].deform_axis = 'POS_X'

def create_road():
    bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0),scale=(1,1,1))
    road_surface = bpy.context.object
    road_surface.name = "Road_Surface"
    road_surface.scale = (.15, 1.5, .15)

    #         Assign material to road surface
    road_material = bpy.data.materials.new(name="Road_Material")
    road_material.diffuse_color = (0,0,0,0)  # Black color
    road_surface.data.materials.append(road_material)

def Infer(IMAGE_FOLDER_NUM,debug=False):
    YOLO3D_output_PATH = f'Data/yolo3d_data/Images_{IMAGE_FOLDER_NUM}.json'
    YOLO2D_output_PATH = f'Data/yolo2d_data/Images_{IMAGE_FOLDER_NUM}.json'

    Lane_Path = f'Data/Lanes/Images_{IMAGE_FOLDER_NUM}_3d.pkl' # Lane data at every frame
    Mesh_path = f'Data/pyMAF/scene_{IMAGE_FOLDER_NUM}/'
    person_pkl_path = 'output.pkl' # Mesh data at every frame
    IMAGE_PATH = f'Data/FinalImages/Images_{IMAGE_FOLDER_NUM}/'

    brakeObjPath = f'Data/yolo2d_data/Images_brake_{IMAGE_FOLDER_NUM}.json'

    # Load jsons data
    with open(YOLO3D_output_PATH, 'r') as f: # Vehicle's 3D data at every frame
        json_data = json.load(f)

    with open(YOLO2D_output_PATH, 'r') as f: # Vehicle's 2D data at every frame
        xy_json_data = json.load(f)
    
    with open(brakeObjPath, 'r') as f: # Vehicle's Brake data at every frame
        brakeData = json.load(f)

    # Importing Lanes
    with open(Lane_Path, 'rb') as f: # Lane data at every frame
        lane_data = pickle.load(f)

    # load pickle data using lane_path
    person_data = joblib.load(Mesh_path+person_pkl_path) # Person's data at every frame

    CAM_LOC = (0, -3, 2.5) # Need to adjust the camera location based on the scene
    
    frame_counter = 0
    for key in json_data.keys():
        frame_counter += 1
        generator = BlenderSceneGenerator()
        Objects_NAMES = [object.name for object in generator.objects]

        if debug:print(Objects_NAMES)
        Objects = []
        Orientations = []
        Locations = []
        Scales = []

        # Every frame detected boxes and classes
        world_2d_boxes = xy_json_data[key]['boxes']
        world_2d_classes = xy_json_data[key]['class_names']
        world_2d_states = xy_json_data[key]['obj_states']
        # Find Depth 
        dep = np.array(getDepthFromImage(zoe,IMAGE_PATH+key))

        if debug:print(f'=== Frame {frame_counter} - {key} Loading , {len(json_data[key])} objects ===')
        pose_details , pose_bbox = get_pose_details(int(key.split('.')[0].split('_')[1]), person_data)

        # explore every box in world_boxes
        done_locs = [] # Visited Locations
        DetText = ''
        
        for box,class_name,state in zip(world_2d_boxes,world_2d_classes,world_2d_states):
            ObjectState = None
            euler_angles, scale, cent = None, None, None

            if debug:print(f'Object : {class_name}')

            # if class_name in vehicle_types:
            #     ObjectState = state
            #     distances = []
            #     # Finding YOLO3D detected vehicle position and orientation
            #     obj_counter = []
            #     for obj_c, obj_det in enumerate(json_data[key]):
            #         # print('Object in  : ', obj_det['Class'])
            #         c_name = obj_det['Class']
            #         if c_name in vehicle_types:
            #             box_2d = obj_det['Box_2d']
            #             x,y = np.mean(box_2d,axis=0)
            #             distances.append((x - box[0])**2 + (y - box[1])**2)
            #             obj_counter.append(obj_c)
            #     if len(distances) == 0 : # No object detected in YOLO3D
            #         # Need to add the class_name object with default orientation, scale, box location
            #         cent = (box[0],box[1])
            #         z_val = dep[int(cent[1]), int(cent[0])]
            #         cent = find_xyz(R, K, cent, z_val)
            #         cent = [cent[0], cent[2], 0]
            #         if cent in done_locs:
            #             continue
            #         done_locs.append(cent)
            #         euler_angles = Blender_rot_scale[class_name]['orientation']
            #         scale = np.array(Blender_rot_scale[class_name]['scale'])
            #         dsts = []
            #         if ObjectState is None or ObjectState == 'Moving':
            #             for box_,c_name in zip(brakeData[key]['boxes'],brakeData[key]['class_names']) :
            #                 if c_name == 'car_BrakeOn':
            #                     x,y = box_[:2]
            #                     dsts.append((x - box[0])**2 + (y - box[1])**2)
            #             if len(dsts) != 0: # Whether the object is in Red state or not
            #                 if min(dsts)<1 :
            #                     class_name += 'Red'
            #     else :
            #         min_index = np.argmin(distances)
            #         obj_det = json_data[key][obj_counter[min_index]]
            #         box_2d = obj_det['Box_2d']
            #         cent = np.mean(box_2d,axis=0)
            #         # cent = box_2d
            #         orien = obj_det['Orientation']
            #         z_val = dep[int(cent[1]), int(cent[0])]
            #         cent = find_xyz(R, K, cent, z_val)
            #         cent = [cent[0], cent[2], 0]

            #         if cent in done_locs:
            #             continue
            #         done_locs.append(cent)

            #         orien, rot = obj_det['Orientation'], obj_det['R']

            #         bird_view_orien = Matrix(((1, 0, 0),
            #                                 (0, 1, 0),
            #                                 (orien[0], orien[1], 0)))
                    
            #         relative_view = bird_view_orien.transposed() @ Matrix(rot)
            #         euler_angles = relative_view.to_euler()
            #         euler_angles += np.array(Blender_rot_scale[class_name]['orientation'])
            #         scale = np.array(Blender_rot_scale[class_name]['scale'])
                # min_index = np.argmin(distances)
                # obj_det = json_data[key][obj_counter[min_index]]
                # box_2d = obj_det['Box_2d']
                # # box_3d = obj_det['Box_3d']
                # orien = obj_det['Orientation']
                # cent = np.mean(box_2d,axis=0)
                # z_val = dep[int(cent[1]), int(cent[0])]
                # cent = find_xyz(R, K, cent, z_val)
                # cent = [cent[0], cent[2], 0]

                # if cent in done_locs:
                #     continue
                # done_locs.append(cent)
                
                # orien, rot = obj_det['Orientation'], obj_det['R']

                # bird_view_orien = Matrix(((1, 0, 0),
                #                             (0, 1, 0),
                #                             (orien[0], orien[1], 0)))
                
                # relative_view = bird_view_orien.transposed() @ Matrix(rot)
                # euler_angles = relative_view.to_euler()
                # euler_angles += np.array(Blender_rot_scale[class_name]['orientation'])
                # scale = np.array(Blender_rot_scale[class_name]['scale'])
            
            if class_name in vehicle_types:
                if debug:print(f'======= * {class_name} * =======')
                ObjectState = state # Stationary or Moving
                distances = []
                # Finding YOLO3D detected vehicle position and orientation
                obj_counter = []
                for obj_c, obj_det in enumerate(json_data[key]):
                    # if debug:print('Object in  : ', obj_det['Class'])
                    c_name = obj_det['Class']
                    if c_name in vehicle_types:
                        box_2d = obj_det['Box_2d']
                        # x,y = box[:2]
                        x,y = np.mean(box_2d,axis=0)
                        distances.append(np.linalg.norm(np.array([box[0],box[1]]) - np.array([x,y])))
                        obj_counter.append(obj_c)
                
                if len(distances) == 0 : # No object detected in YOLO3D
                    # Need to add the class_name object with default orientation, scale, box location
                    cent = (box[0],box[1])
                    z_val = dep[int(cent[1]), int(cent[0])]
                    cent = find_xyz(R, K, cent, z_val)
                    cent = [cent[0], cent[2], 0]
                    if cent in done_locs:
                        continue
                    done_locs.append(cent)
                    euler_angles = Blender_rot_scale[class_name]['orientation']
                    scale = np.array(Blender_rot_scale[class_name]['scale'])
                    distances = []
                    if ObjectState is None or ObjectState == 'Moving':
                        if class_name.lower() in ['car', 'truck', 'suv', 'pickup truck']:
                            for box_brake,c_name in zip(brakeData[key]['boxes'],brakeData[key]['class_names']) :
                                if c_name == 'car_BrakeOn':
                                    
                                    distances.append(np.linalg.norm(np.array([box_brake[0],box_brake[1]]) - np.array([box[0],box[1]])))

                        if len(distances) != 0:
                            class_name += 'Red'
                    del distances
                    
                else :
                    min_index = np.argmin(distances)
                    obj_det = json_data[key][obj_counter[min_index]]
                    # box_2d = box[:2]
                    box_2d = obj_det['Box_2d']
                    # box_3d = obj_det['Box_3d']
                    orien = obj_det['Orientation']
                    cent = np.mean(box_2d,axis=0)
                    # cent = box[:2]
                    z_val = dep[int(cent[1]), int(cent[0])]

                    # Check if the object's rear light is on or off
                    if ObjectState is None or ObjectState == 'Moving':
                        distances = []
                        if class_name.lower() in ['car', 'truck', 'suv', 'pickup truck']:
                            for box_brake,c_name in zip(brakeData[key]['boxes'],brakeData[key]['class_names']) :
                                if c_name == 'car_BrakeOn':
                                    # box_2d = obj_det['Box_2d']
                                    distances.append(np.linalg.norm(np.array([box_brake[0],box_brake[1]]) - np.mean(box_2d,axis=0)))
                        
                        if len(distances) != 0:
                            class_name += 'Red'
                        del distances
                    # if debug:print(cent)
                    cent = find_xyz(R, K, cent[:2], z_val)
                    cent = [cent[0], cent[2], 0]

                    if cent in done_locs:
                        continue
                    done_locs.append(cent)
                    orien, rot = obj_det['Orientation'], obj_det['R']

                    bird_view_orien = Matrix(((1, 0, 0),
                                                (0, 1, 0),
                                                (orien[0], orien[1], 0)))
                    
                    relative_view = bird_view_orien.transposed() @ Matrix(rot)
                    euler_angles = relative_view.to_euler()
                    euler_angles += np.array(Blender_rot_scale[class_name]['orientation'])
                    scale = np.array(Blender_rot_scale[class_name]['scale'])
                    # del cent,z_val
                if debug:print(f'======= * {class_name} DONE * =======')
            
            elif class_name == 'person' : # Persons working Perfectly , Importing meshes below
                continue
            
            elif class_name in traffic_light_types : # Traffic Lights working Perfectly
                if debug:print(f'======= * {class_name} * =======')
                distances = []
                for obj_c, obj_det in enumerate(json_data[key]):
                    c_name = obj_det['Class']
                    box_2d = obj_det['Box_2d']
                    x,y = np.mean(box_2d, axis=0)
                    distances.append(np.linalg.norm(np.array([box[0],box[1]]) - np.array([x,y])))
                if len(distances) == 0:
                    cent = (box[0],box[1])
                    z_val = dep[int(cent[1]), int(cent[0])]
                    cent = find_xyz(R, K, cent, z_val)
                    cent = [cent[0], cent[2], 0]
                    if cent in done_locs:
                        continue
                    done_locs.append(cent)
                    euler_angles = Blender_rot_scale[class_name]['orientation']
                    scale = np.array(Blender_rot_scale[class_name]['scale'])
                    continue
                min_ind = np.argmin(distances)
                del distances
                # if debug:print(min_ind,json_data[key][min_ind]['Class'])
                obj_det = json_data[key][min_ind]
                c_name = class_name
                box_2d = obj_det['Box_2d']
                orien = obj_det['Orientation']
                cent = np.mean(box_2d, axis=0)

                z_val = dep[int(cent[1]), int(cent[0])]
                cent = find_xyz(R, K, cent, z_val)
                # if debug:if debug:print('Traffic light 1 : ',cent)
                cent = [cent[0], cent[2], abs(cent[1])+CAM_LOC[-1]] # x,y,z locations Wrong locations
                if cent in done_locs:
                    continue
                done_locs.append(cent)
                # del cent,z_val,box_2d,orien
                euler_angles = Blender_rot_scale[class_name]['orientation']
                scale = np.array(Blender_rot_scale[class_name]['scale'])
                if debug:print(f'======= * {class_name} DONE * =======')
            elif class_name == 'road sign' : # check is detected object is having any number or not
                if debug:print(f'======= * {class_name} * =======')
                image = cv2.imread(IMAGE_PATH+key)
                x1,y1,x2,y2 = int(box[0]-box[2]/2),int(box[1]-box[3]/2),int(box[0]+box[2]/2),int(box[1]+box[3]/2)
                image = image[y1:y2,x1:x2]
                del x1,y1,x2,y2
                DetText, num_bool = findText(reader,image)
                # if debug:print(DetText, num_bool)
                if num_bool:
                    cent = (box[0],box[1])
                    z_val = dep[int(cent[1]), int(cent[0])]
                    cent = find_xyz(R, K, cent, z_val)
                    cent = [cent[0], cent[2], 0]
                    if cent in done_locs:
                        continue
                    done_locs.append(cent)
                    class_name = f'parking meter'
                    # del cent,z_val
                    euler_angles = Blender_rot_scale[class_name]['orientation']
                    scale = np.array(Blender_rot_scale[class_name]['scale'])
                elif 'hump' in DetText.lower() :
                    class_name = 'hump'
                    cent = (box[0],box[1])
                    z_val = dep[int(cent[1]), int(cent[0])]
                    cent = find_xyz(R, K, cent, z_val)
                    cent = [cent[0]-3, cent[2]-3, 0]
                    if cent in done_locs:
                        continue
                    done_locs.append(cent)
                    # del cent,z_val
                    euler_angles = Blender_rot_scale[class_name]['orientation']
                    scale = np.array(Blender_rot_scale[class_name]['scale'])
                del image
                if debug:print(f'======= * {class_name} DONE * =======')
            else: # Check if class_name is speed limit sign or not
                if debug:print(f'======= * {class_name} * =======')
                cent = (box[0],box[1])
                z_val = dep[int(cent[1]), int(cent[0])]
                cent = find_xyz(R, K, cent, z_val)
                cent = [cent[0] , cent[2]-3 , 0]
                if cent in done_locs:
                    continue
                done_locs.append(cent)
                euler_angles = Blender_rot_scale[class_name]['orientation']
                scale = np.array(Blender_rot_scale[class_name]['scale']) #* scale_fac
                # del cent,z_val
                if debug:print(f'======= * {class_name} DONE * =======')
            if class_name in YOLOCLASSES_TO_BLENDER :
                if debug:print(f'======= * Blender Loading * =======')
                if class_name != 'person':
                    blend_mesh_name = YOLOCLASSES_TO_BLENDER[class_name]
                    # if debug:if debug:print(blend_mesh_name)
                    ind = next(i for i, s in enumerate(Objects_NAMES) if s.split('.')[0].startswith(blend_mesh_name.split('.')[0]))
                    if debug:print(f'{blend_mesh_name} - {generator.objects[ind]} Loaded ... ')
                    if ObjectState == 'Stationary' and class_name in vehicle_types:
                        obj = generator.objects[ind]
                        obj = addTexture(obj)
                        Objects.append(obj)
                    else :
                        Objects.append(generator.objects[ind])
                    Orientations.append(euler_angles)
                    Locations.append(cent)
                    Scales.append(scale)
                    if debug:print(f'frame_counter : {frame_counter} , Object : {blend_mesh_name} Added ...')
                if debug:print(f'======= * Blender Loaded* =======')
        # Importing Person Objects
        
        for i in range(len(pose_details)):
            obj_path = pose_details[i]
            obj_bbox = pose_bbox[i]
            center = (obj_bbox[0], obj_bbox[1])
            z_val = dep[int(center[1]), int(center[0])]
            center = find_xyz(R, K, center, z_val)
            center = [center[0], center[2], .5]
            euler_angles = Blender_rot_scale['person']['orientation']
            scale = np.array(Blender_rot_scale['person']['scale'])
            bpy.ops.wm.obj_import(filepath=Mesh_path+obj_path)
            imported_obj = bpy.context.selected_objects[0]
            # if debug:print(center)
            imported_obj.location = center
            # imported_obj.rotation_euler = euler_angles
            # imported_obj.scale = scale
        del pose_details, pose_bbox, dep

        # # Importing Road
        create_road()

        lane = lane_data[key]
        lane_3d_pts , lane_3d_class = lane['final_lanes'], lane['final_lane_classes']
        del lane
        for pt,cl in zip(lane_3d_pts,lane_3d_class):
            curve_obj = create_bezier_curve_from_points(pt)
            if cl == "solid-line":
                create_lane_markings_by_curve_length(curve_obj, lane_width=1, lane_length=10, gap_length=0, num_lanes=30)
            else:
                create_lane_markings_by_curve_length(curve_obj, lane_width=1, lane_length=10, gap_length=1, num_lanes=30)

        if debug:print('========= Objects Loading into the frame... ==========')
        generator.load_objects_into_frame(Objects, Orientations, Locations, Scales, frame_counter, CAM_LOC)
        # if ObjectState == 'Stationary' and class_name in vehicle_types:
        #     generator.load_objects_into_frame(Objects, Orientations, Locations, Scales, frame_counter, CAM_LOC, ObjectState=ObjectState)
        # else :
        #     generator.load_objects_into_frame(Objects, Orientations, Locations, Scales, frame_counter, CAM_LOC)
        if debug:print('========= Objects Loaded into the frame... ==========')
        del Objects, Orientations, Locations, Scales
        if debug:print('========= Frame Rendering ==========')
        generator.render_frame(f"Outputs/BlenderImages/{IMAGE_FOLDER_NUM}/", frame_name=key , frame_num=frame_counter)
        if debug:print('========= Frame Rendered ==========')
        del done_locs
        del generator

    # Collision prediction
    # TODO :
    # 1. Find Object's distance and direction of the objects.
    # 2. If they are directing the same direction then make Object to Red state.
    #########

# Take arguments from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Generate Blender scenes')
    # parser.add_argument('--image_folder_num', type=int, default=0, help='Image folder number')
    # add debug True or False argument
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    # parser.add_argument('--num1', type=str, default=50, help='Start Index')
    # parser.add_argument('--num2', type=str, default=50, help='End Index')
    return parser.parse_args()

def main() :
    args = parse_args()
    # IMAGE_FOLDER_NUM = args.image_folder_num
    debug = args.debug
    # Infer(IMAGE_FOLDER_NUM, debug=debug)
    # args.path = f'Data/Video_{IMAGE_FOLDER_NUM}/'
    for i in range(1,14) : # 1 to 3 scenes
        Infer(i ,debug=debug)
    # break

if __name__ == '__main__' :
    main()

# python main.py --image_folder_num 1 --debug True

##############################################################################################################
# TODO :
# 1. Need to add vehicles if len(distances) = 0 -> Done
# 2. check whether object break on or off -> Done
# 3. Find object's state stationary or moving -> Done
# 4. Add SpeedHump -> Done
# 5. Collision prediction
##############################################################################################################