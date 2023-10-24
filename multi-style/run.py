import sys
sys.path.append('.')

import cv2
from source.cartoonize import Cartoonizer
import os
import argparse



from pathlib import Path
from itertools import chain
from glob import glob


def get_path_images(path_in: Path):
    allowed = ['*.png', '*.jpg', '*.jpeg', '*.jfif', '*.gif', '*.webp']
    path_in_images = list(chain.from_iterable([
        glob(str(path_in/'**'/t), recursive=True)
        for t in allowed
    ]))
    return path_in_images


def process(args):

    style = args.style
    if style == "anime":
        algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon_compound-models')

    elif style == "3d":
        algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon-3d_compound-models')

    elif style == "handdrawn":
        algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon-handdrawn_compound-models')

    elif style == "sketch":
        algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon-sketch_compound-models')

    elif style == "artstyle":
        algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon-artstyle_compound-models')

    else:
        print('no such style %s' % style)
        return 0

    args.path_out.mkdir(parents=True, exist_ok=True)

    for path_image in get_path_images(args.path_in):
        # img = cv2.imread('input.png')[..., ::-1]
        img = cv2.imread(path_image)[..., ::-1]
        result = algo.cartoonize(img)
        # cv2.imwrite('result1_%s.png'%style, result)
        path_image = Path(path_image)
        cv2.imwrite(str(args.path_out/f'{path_image.stem}_{style}{path_image.suffix}'), result)

    print('finished!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='anime')
    parser.add_argument('--path-in', type=Path)
    parser.add_argument('--path-out', type=Path)
    args = parser.parse_args()

    process(args)
