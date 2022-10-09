import sys
sys.path.append('.')

import cv2
from source.cartoonize import Cartoonizer
import os
import argparse


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

    img = cv2.imread('input.png')[..., ::-1]
    result = algo.cartoonize(img)
    cv2.imwrite('result1_%s.png'%style, result)

    print('finished!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='anime')
    args = parser.parse_args()

    process(args)



