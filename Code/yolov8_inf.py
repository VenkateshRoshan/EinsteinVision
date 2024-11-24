
from ultralytics import YOLO
import argparse
import cv2
from matplotlib import pyplot as plt
import os
import json
from PIL import Image
from mmflow.apis import inference_model, init_model
import mmcv
import numpy as np

def load_model():
    model = YOLO('models/yolov8x.pt')
    # model = YOLO('yolov8n-pose.pt')
    return model    

def load_world_model(classes=None) :
    model = YOLO('models/yolov8x-worldv2.pt')
    classes = ["car", "suv", "pickup truck" , "truck", "sedan", "person", "green traffic light", 
            "red traffic light", "yellow traffic light", "traffic cone", "speed limit sign", "bicycle", 
            "road sign", "stop sign", "speed breaker", "speed hump", "traffic cylinder" , 'trash can']
    model.set_classes(classes)
    return model

def compute_fundamental_matrix_sift(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Match keypoints between the two images
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Extract corresponding keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute fundamental matrix using RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    
    return fundamental_matrix

def compute_intersection_point(epipolar_line, width, height):
    """
    Compute intersection point of epipolar line with image boundaries.
    """
    x0, y0, z0 = epipolar_line
    if z0 != 0:
        x_intersection = 0
        y_intersection = int((-x0 * x_intersection - z0) / y0)
        if 0 <= y_intersection < height:
            return x_intersection, y_intersection
        else:
            y_intersection = height - 1
            x_intersection = int((-y0 * y_intersection - z0) / x0)
            if 0 <= x_intersection < width:
                return x_intersection, y_intersection
            else:
                return None, None
    else:
        return None, None

def calculate_movement(image1, image2, bbox, flow, F):
    """
    This function classifies car movement in image2 within the bounding box (bbox) based on flow between image1 and image2 using Sampson distance and fundamental matrix.

    Args:
        image1 (np.ndarray): First image (grayscale).
        image2 (np.ndarray): Second image (grayscale).
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        flow (np.ndarray): Flow image representing displacement vectors.
        F (np.ndarray): Fundamental matrix.

    Returns:
        bool: True if moving, False if static.
    """

    # Extract flow vectors and image dimensions within bounding box
    flow_subset = flow[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    height, width = flow_subset.shape[:2]

    # Select salient points within the bounding box using corner detection
    corners = cv2.goodFeaturesToTrack(image1[bbox[1]:bbox[3], bbox[0]:bbox[2]], maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is None:
        return False
    points1 = np.int0(corners).reshape(-1, 2)

    # Initialize list to store Sampson distances
    sampson_distances = []

    # Iterate through each point in the first frame within the bounding box
    for point1 in points1:
        x1, y1 = point1

        # Extract flow vector for this point
        flow_vec = flow_subset[y1, x1]

        # Expected displacement based on flow
        expected_displacement = flow_vec

        # Reprojection using fundamental matrix
        x1_homog = np.array([x1, y1, 1])
        epipolar_line = np.dot(F, x1_homog)
        x2_expected, y2_expected = compute_intersection_point(epipolar_line, width, height)

        # Check if expected reprojected point is within image bounds
        if x2_expected is not None and 0 <= x2_expected < width and 0 <= y2_expected < height:
            actual_displacement = [image2[y2_expected, x2_expected] - image1[y1, x1]]

            # Sampson distance calculation
            sampson_distance = np.linalg.norm(expected_displacement - actual_displacement) ** 2 / (expected_displacement[0] ** 2)
            sampson_distances.append(sampson_distance)

    # Classification based on average Sampson distance and threshold
    threshold = 2 # Adjust this based on your application and expected flow values
    print(len(sampson_distances))
    if len(sampson_distances) == 0:
        return True
	
    avg_sampson_distance = np.mean(sampson_distances)
    print("Average Sampson Distance: ", avg_sampson_distance)
    if avg_sampson_distance < threshold :
        return True  # Moving
    else:
        return False  # Static

def get_keypoints(image_path,show=False):
    model = YOLO('models/yolov8x-pose.pt')
    results = model(image_path)
    keypoints_ = results[0].keypoints.xy.cpu().numpy()
    if show:
        for keypoints in keypoints_ :
            # print(keypoints.data)
            # exit()
            print('====================================')
            print(keypoints)
            print('====================================')
            # read keypoints and plot them
            img = cv2.imread(image_path)
            for x,y in keypoints :
                # print(x,y)
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            plt.imshow(img)
            plt.show()

    return keypoints_

# def 
    
def predict_image(model, img_path,show_info=False):
    results = model.predict(img_path)
    boxes_total = results[0].boxes.xywh.cpu().numpy()
    classes_total = results[0].boxes.cls.cpu().numpy()
    scores_total = results[0].boxes.conf.cpu().numpy()
    total_labels = results[0].names
    classes_names = []
    for i in range(len(classes_total)):
        classes_names.append(total_labels[classes_total[i]])
    
    if show_info:
        print("====================================")
        print(f"{img_path} Predictions")
        print("Boxes: ", boxes_total)
        print("Classes: ", classes_total)
        print("Scores: ", scores_total)
        print("Classes Names: ", classes_names)
        print("====================================")
    return results, boxes_total, classes_total, scores_total, classes_names
        
# plot bounding boxes on image and label them
def plot_boxes(image_path , boxes , class_names , show = True):
    image = cv2.imread(image_path)
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        # x , y are middle co-ords and w, h are width and height
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_names[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite("output.jpg", image)
    if show :
        plt.imshow(image)
        plt.show()

def calculateMovement(PREV_IMAGE_PATH,CUR_IMAGE_PATH,boxes):
    config_file = 'models/raft_8x2_100k_flyingthings3d_sintel_368x768.py'
    checkpoint_file = 'models/raft_8x2_100k_flyingthings3d_sintel_368x768.pth'
    device = 'cuda:0'

    model = init_model(config_file, checkpoint_file, device=device)

    img1 = cv2.imread(PREV_IMAGE_PATH)
    img2 = cv2.imread(CUR_IMAGE_PATH)

    output = inference_model(model,img1, img2)
    flow_map = np.uint8(mmcv.flow2rgb(output) * 255.)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    fundamental_matrix  = compute_fundamental_matrix_sift(img1, img2)

    obj_states = []
    for i in range(len(boxes)):
        x = int(boxes[i][0])
        y = int(boxes[i][1])
        w = int(boxes[i][2])
        h = int(boxes[i][3])
        
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
        obj_states.append(label_moving)
        
    return obj_states

# write arg parser to take image path
def arg_parser():
    parser = argparse.ArgumentParser(description='Predict on an image')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--json_path', type=str, help='Path to the json file to save keypoints and bounding boxes')
    args = parser.parse_args()
    return args
    
def main():
    # model = load_model()
    model = load_world_model()
    # image_path= "eval/image_10_sort/frame_730.jpg"
    # Image_Path = 'eval/image_10_sort/'
    args = arg_parser()

    for i in range(1,14) :
        # send every image to predict_image and get keypoints and save it in a json file as image_path is key
        Frame_2D_Data = {}
        
        Image_Path = args.image_path+f'Images_{i}/'
        json_path = args.json_path+f'Images_{i}.json'

        PREV_IMAGE_PATH, CUR_IMAGE_PATH = None, None
        obj_states = None
        k = 0
        for image_path in sorted(os.listdir(Image_Path)) :
            if image_path.endswith('.jpg'):
                key = image_path.split('/')[-1]
                
                # print(key)
                Frame_2D_Data[key] = {}
                image_path = os.path.join(Image_Path,image_path)
                results, boxes, classes, scores, class_names = predict_image(model, image_path)
                if k != 0 :
                    PREV_IMAGE_PATH = CUR_IMAGE_PATH
                    CUR_IMAGE_PATH = Image_Path+image_path
                    obj_states = calculateMovement(PREV_IMAGE_PATH,CUR_IMAGE_PATH,boxes)
                else :
                    CUR_IMAGE_PATH = Image_Path+image_path
                    obj_states = None
                # print(boxes , classes , class_names)
                Frame_2D_Data[key]['boxes'] = boxes.tolist()
                Frame_2D_Data[key]['class_names'] = class_names
                if obj_states is None:
                    Frame_2D_Data[key]['obj_states'] = ['Moving' for i in range(len(class_names))]
                else :
                    Frame_2D_Data[key]['obj_states'] = obj_states

                k += 1

        with open(json_path, 'w') as f:
            json.dump(Frame_2D_Data, f , indent=4)

        # break

if __name__ == '__main__':
    main()


# python yolov8_inf.py --image_path Data/FinalImages/Images_1/ --json_path Data/