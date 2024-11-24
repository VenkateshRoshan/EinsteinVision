
import argparse
import easyocr
import cv2
import numpy as np

def run(reader,image) :

    # reader = easyocr.Reader(['en'], gpu=True)
    # print(dir(reader))
    text = reader.readtext(image,detail=1)
    # print(text)
    boxes = []
    texts = []
    probs = []
    Text = ''
    if text == []:
        return Text, False
    for box, text, prob in text:
        boxes.append(box)
        texts.append(text)
        probs.append(prob)
    max_ind = np.argmax(probs)
    text = texts[max_ind]
    # print(f'Text Numeric or Not: {text} : {text.isnumeric()}')
    if text.isnumeric():
        return text, True
    else :
        text_ = ''
        for t in texts:
            text_ += t
        # print(text_)
        return text_.lower(), False

def arg_parser():
    parser = argparse.ArgumentParser(description='Predict on an image')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parser()
    run(args.image_path)