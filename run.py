
import cv2
from source.cartoonize import Cartoonizer
import os

def process():

    algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon_compound-models')
    img = cv2.imread('input.png')[...,::-1]

    result = algo.cartoonize(img)

    cv2.imwrite('res.png', result)
    print('finished!')




if __name__ == '__main__':
    process()



