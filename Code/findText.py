
import numpy as np
import json
import cv2
import argparse
from ocrInference import run as ocr_run
from matplotlib import pyplot as plt

def findText(image_path,class_names, boxes):
    for class_name, box in zip(class_names, boxes):
        if class_name == 'road sign' :
            image = cv2.imread(image_path)
            x,y,w,h = box
            x1,y1,x2,y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            image = image[y1:y2, x1:x2]
            text,num_bool = ocr_run(image_path)
            if num_bool:
                return text,num_bool
            else:
                return text,False
    return '',False

# def findText(image_path, json_path):
#     # load json
#     SPEED_LIMIT_SVG_PATH = 'Assets/Pictures/Speed_Limit_blank_sign.png'
#     with open(json_path) as f:
#         json_data = json.load(f)
#     for key in json_data.keys() :
#         # if key=='frame_01921.jpg' or  key=='frame_01911.jpg':
#         # if key=='frame_00291.jpg':
#         class_names = json_data[key]['class_names']
#         boxes = json_data[key]['boxes']
#         # print(f'classes: {class_names}')
#         # print(f'boxes: {boxes}')
#         for class_name, box in zip(class_names, boxes):
#             if class_name == 'road sign' :
#                 image = cv2.imread(image_path+key)
#                 x,y,w,h = box
#                 x1,y1,x2,y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
#                 image = image[y1:y2, x1:x2]
#                 text,num_bool = ocr_run(image_path+key)
#                 return num_bool
#             #     print(f'Text: {text}')
#             #     if num_bool:

#             #         image = cv2.imread(SPEED_LIMIT_SVG_PATH)

#             #         # Using cv2.putText()
#             #         new_image = cv2.putText(
#             #             img = image,
#             #             text = text,
#             #             org = (750, 2700),
#             #             fontFace = cv2.FONT_HERSHEY_SIMPLEX,
#             #             fontScale = 30.0,
#             #             color = (0,0,0),
#             #             thickness = 100
#             #             )

#             #         cv2.imwrite('speed_sign.jpg', new_image)
#             #         # plt.imshow(new_image)
#             #         # plt.show()
                    
#             # break
    

def arg_parser():
    parser = argparse.ArgumentParser(description='Predict on an image')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--json_path', type=str, help='yolo2d')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    findText(args.image_path, args.json_path)

# Images_9 , frame_01911.jpg