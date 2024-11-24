from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow
import os.path as osp
import cv2 
import sys 
sys.path.append("../")
sys.path.append("../Einstein-Vision")
sys.path.append("../Einstein-Vision/utilities")
from utilities.cv2_utilities import *
from ultralytics import YOLO
import mmcv
import numpy as np

def load_model():
    # classes
    classes = ["car", "suv", "pickup truck", "truck", "sedan", "person", "bicycle"]
    model = YOLO('yolov8x-worldv2.pt')
    model.set_classes(classes)
    return model

def predict_image(model, img_path):
    results = model.predict(img_path)
    results[0].show()
    boxes_total = results[0].boxes.xywh.cpu().numpy()
    classes_total = results[0].boxes.cls.cpu().numpy()
    scores_total = results[0].boxes.conf.cpu().numpy()
    total_labels = results[0].names
    classes_names = []
    for i in range(len(classes_total)):
        classes_names.append(total_labels[classes_total[i]])
        
    # print("====================================")
    # print("Predictions")
    # print("Boxes: ", boxes_total)
    # print("Classes: ", classes_total)
    # print("Scores: ", scores_total)
    # print("Classes Names: ", classes_names)
    # print("====================================")
    return results, boxes_total, classes_total, scores_total, classes_names

def get_movement_classification(img1, img2, boxes_total):
    # Global variables for Model paths
    config_file = 'models/raft_8x2_100k_flyingthings3d_sintel_368x768.py'
    checkpoint_file = 'models/raft_8x2_100k_flyingthings3d_sintel_368x768.pth'
    device = 'cuda:0'
    
    # init a model
    model = init_model(config_file, checkpoint_file, device=device)
    # inference the demo image
    output = inference_model(model,img1, img2)
    
    flow_map = np.uint8(mmcv.flow2rgb(output) * 255.)
    # Find point correspondences between two images
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    fundamental_matrix  = compute_fundamental_matrix_sift(img1, img2)
    
    labels_list = []
    for i in range(len(boxes_total)):
        x = int(boxes_total[i][0])
        y = int(boxes_total[i][1])
        w = int(boxes_total[i][2])
        h = int(boxes_total[i][3])
        
        
        x_min = int(x-w//2)
        y_min = int(y-h//2)
        x_max = int(x+w//2)
        y_max = int(y+h//2)
        
        bbox = (x_min, y_min, x_max, y_max)
        flow_field = flow_map
        sampson_distance_out = calculate_movement(img1_gray, img2_gray, bbox, flow_field, fundamental_matrix)
        
        if sampson_distance_out == True:
            label_moving = "Moving"
        else:
            label_moving = "Stationary"
        labels_list.append(label_moving)
        
    return labels_list