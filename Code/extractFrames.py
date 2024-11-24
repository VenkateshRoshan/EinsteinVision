
import cv2
import argparse
import os
from tqdm import tqdm

# Extract frames from video
def extractFrames(pathIn, pathOut, skip=5):
    # Path to video file
    cap = cv2.VideoCapture(pathIn)
    count = 0
    ImgCounter = 0

    # Check if video file is opened
    if (cap.isOpened()==False):
        print("Error opening video file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # print('Read %d frame: ' % count, ret)
            print(f'\r Extracting frame {count+1}', end='')
            
            # Save the extracted frames
            if count % skip == 0:
                ImgCounter += 1
                # save frame name with %4d and counter as frame number
                cv2.imwrite(os.path.join(pathOut, "frame_%05d.jpg" % (ImgCounter)), frame)
            count += 1
        else:
            break

    cap.release()

def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--image_path', type=str, help='Path to save extracted frames')
    # add an argument to skip images
    parser.add_argument('--skip_images', type=int, default=5, help='Number of images to skip')
    return parser.parse_args()

def main() :
    args = parse_args()
    v_path = args.video_path # Sequences
    i_path = args.image_path # Data/Images
    skip = int(args.skip_images)
    for i in tqdm(range(1, 14)):
        video_path = f'{v_path}/scene{i}/Undist/'
        image_path = f'{i_path}/Images_{i}/' 
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        videoPath = None
        for video_path_ in os.listdir(video_path):
            if 'front' in video_path_ :
                videoPath = f'{video_path}{video_path_}'
        
        # print(f'Extracting frames from {videoPath} to {image_path}')
        extractFrames(videoPath, image_path, skip)
        # print('\n')

if __name__ == "__main__" :
    main()

# python extractFrames.py --video_path Sequences --image_path Data/Images --skip_images 10