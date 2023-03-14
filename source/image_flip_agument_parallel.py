import oss2
import argparse
import cv2
import glob
import os
import tqdm
import numpy as np
# from .utils import get_rmbg_alpha, get_img_from_url,reasonable_resize,major_detection,crop_img
import tqdm
import urllib
import random
from multiprocessing import Pool


parser = argparse.ArgumentParser(description="process remove bg result")
parser.add_argument("--data_dir", type=str, default="", help="Path to images.")
parser.add_argument("--save_dir", type=str, default="", help="Path to save images.")
args = parser.parse_args()


args.save_dir = os.path.join(args.data_dir, 'total_flip')
form = 'single'


def flipImage(image):
    new_image = cv2.flip(image, 1)
    return new_image

def all_file(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg' or extend == '.JPG':
                L.append(os.path.join(root, file))
    return L


paths = all_file(args.data_dir)


def process(path):

    print(path)
    outpath = args.save_dir+path[len(args.data_dir):]
    if os.path.exists(outpath):
        return

    sub_dir = os.path.dirname(outpath)
    # print(sub_dir)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir,exist_ok=True)

    img = cv2.imread(path, -1)
    h, w, c = img.shape
    if form == "pair":
        imga = img[:, :int(w / 2), :]
        imgb = img[:, int(w / 2):, :]
        imga = flipImage(imga)
        imgb = flipImage(imgb)
        res = cv2.hconcat([imga, imgb])  # 水平拼接

    else:
        res = flipImage(img)

    cv2.imwrite(outpath, res)
    print('save %s' % outpath)





if __name__ == "__main__":
    # main(args)
    pool = Pool(100)
    rl = pool.map(process, paths)
    pool.close()
    pool.join()