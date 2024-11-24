

import cv2
import os
import argparse
from tqdm import tqdm

def images_to_video(image_folder, video_name, fps):

    # read image from image folder
    images = []
    for img in sorted(os.listdir(image_folder)) :
        images.append(img)

    # print(f'Number of images: {len(images)}')
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    # print(f'Height: {height}, Width: {width}, Layers: {layers}') # 960,1280,3

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()

# Take arguments to get Image Path and output video name

def arg_parser():
    parser = argparse.ArgumentParser(description='Convert images to video')
    parser.add_argument('--image_folder', type=str, help='Path to the image folder')
    parser.add_argument('--video_name', type=str, help='Name of the output video')
    parser.add_argument('--fps', type=int, default=10 , help='Frames per second')
    parser.add_argument('--num', type=int, default=1 , help='Number of the scene')
    args = parser.parse_args()
    return args
    
def main() :
    args = arg_parser()

    # for i in tqdm(range(1,14)) :
    i = args.num
    image_folder = f'{args.image_folder}/{i}/'
    video_name = f'{args.video_name}/scene_{i}.mp4'
    if not os.path.exists(args.video_name):
        os.makedirs(args.video_name)
    images_to_video(image_folder, video_name, args.fps)
        # print(f'\rVideo saved at {video_name}')

    # images_to_video(args.image_folder, args.video_name, args.fps)

if __name__ == "__main__":
    main()

# python generateVideo.py --image_folder Data/Images/ --video_name Outputs/OrgVideos/ --fps 10
# python generateVideo.py --image_folder Outputs/BlenderImages/ --fps 5 --num 1 --video_name Outputs/BlenderVideos