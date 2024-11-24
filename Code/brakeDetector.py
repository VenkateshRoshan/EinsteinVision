
import cv2
import json
import argparse
from ultralytics import YOLO
import os
from yolov8_inf import *

def load_model():
    model = YOLO('models/brakeDetection.pt')
    # model = YOLO('yolov8n-pose.pt')
    classes=['car brake on', 'car brake off']
    # model.set_classes(classes)
    return model

# write arg parser to take image path
def arg_parser():
    parser = argparse.ArgumentParser(description='Predict on an image')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--json_path', type=str, help='Path to the json file to save keypoints and bounding boxes')
    args = parser.parse_args()
    return args
    
def main():
    # model = load_model()
    model = load_model()
    # image_path= "eval/image_10_sort/frame_730.jpg"
    # Image_Path = 'eval/image_10_sort/'
    args = arg_parser()

    for i in range(1,14) :
        # send every image to predict_image and get keypoints and save it in a json file as image_path is key
        Frame_2D_Data = {}
        
        Image_Path = args.image_path+f'Images_{i}/'
        json_path = args.json_path+f'Images_brake_{i}.json'

        for image_path in sorted(os.listdir(Image_Path)) :
            if image_path.endswith('.jpg'):
                key = image_path.split('/')[-1]
                # print(key)
                Frame_2D_Data[key] = {}
                image_path = os.path.join(Image_Path,image_path)
                results, boxes, classes, scores, class_names = predict_image(model, image_path)
                # print(boxes , classes , class_names)
                Frame_2D_Data[key]['boxes'] = boxes.tolist()
                Frame_2D_Data[key]['class_names'] = class_names

                # keypoints = get_keypoints(image_path)
                # if keypoints is not None or len(keypoints) != 0 :
                #     Frame_2D_Data[key]['keypoints'] = keypoints.tolist()
                # else:
                #     Frame_2D_Data[key]['keypoints'] = []

        # print(Frame_2D_Data)

        with open(json_path, 'w') as f:
            json.dump(Frame_2D_Data, f , indent=4)

if __name__ == '__main__':
    main()

# python brakeDetector.py --image_path eval/image_10_sort/ --json_path Frame_2D_json_path.json