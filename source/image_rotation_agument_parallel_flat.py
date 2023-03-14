import oss2
import argparse
import cv2
import glob
import os
import tqdm
import numpy as np
import tqdm
import urllib
import random
from multiprocessing import Pool


parser = argparse.ArgumentParser(description="process remove bg result")
parser.add_argument("--data_dir", type=str, default="", help="Path to images.")
parser.add_argument("--save_dir", type=str, default="", help="Path to save images.")
args = parser.parse_args()


args.save_dir = os.path.join(args.data_dir, 'total_rotate')
form = 'single'

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir,exist_ok=True)

def all_file(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg':
                L.append(os.path.join(root, file))
    return L

def rotateImage(image, angle):
    row,col,_ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row), borderMode=cv2.BORDER_REFLECT)
    return new_image


paths = all_file(args.data_dir)


def process(path):

    if 'total_scale' in path:
        return

    outpath = args.save_dir + path[len(args.data_dir):]
    sub_dir = os.path.dirname(outpath)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir, exist_ok=True)

    img0 = cv2.imread(path, -1)
    h, w, c = img0.shape
    img = img0[:, :, :3].copy()
    if c == 4:
        alpha = img0[:, :, 3]
        mask = alpha[:, :, np.newaxis].copy() / 255.
        img = (img * mask + (1 - mask) * 255)

    imgb = None
    imgc = None
    if form is 'single':
        imga = img
    elif form is 'pair':
        imga = img[:, :int(w / 2), :]
        imgb = img[:, int(w / 2):, :]
    elif form is 'tuple':
        imga = img[:, :int(w / 3), :]
        imgb = img[:, int(w / 3): int(w * 2 / 3), :]
        imgc = img[:, int(w * 2 / 3):, :]

    angles = [ random.randint(-10, 0), random.randint(0, 10)]

    for angle in angles:

        imga_r = rotateImage(imga, angle)
        if form is 'single':
            res = imga_r
        elif form is 'pair':
            imgb_r = rotateImage(imgb, angle)
            res = cv2.hconcat([imga_r, imgb_r])  # 水平拼接
        else:
            imgb_r = rotateImage(imgb, angle)
            imgc_r = rotateImage(imgc, angle)
            res = cv2.hconcat([imga_r, imgb_r, imgc_r])  # 水平拼接

        cv2.imwrite(outpath[:-4]+'_'+str(angle)+'.png', res)
        print('save %s'% outpath)



if __name__ == "__main__":
    # main(args)
    pool = Pool(100)
    rl = pool.map(process, paths)
    pool.close()
    pool.join()