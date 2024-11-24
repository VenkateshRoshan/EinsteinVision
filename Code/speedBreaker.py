

from ultralytics import YOLO
import argparse
import cv2
from matplotlib import pyplot as plt
import os
import json
from PIL import Image

def load_world_model(classes=None) :
    model = YOLO('models/yolov8x-worldv2.pt')
    classes = ['speed breaker','speed hump']
    model.set_classes(classes)
    return model
    
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

# python yolov8_inf.py --image_path eval/image_10_sort/ --json_path Frame_2D_json_path.json