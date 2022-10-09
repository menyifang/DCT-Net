import cv2, argparse
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def process(args):
    style = args.style
    print('choose style %s'%style)
    if style == "anime":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,
                               model='damo/cv_unet_person-image-cartoon_compound-models')
    elif style == "3d":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,
                               model='damo/cv_unet_person-image-cartoon-3d_compound-models')
    elif style == "handdrawn":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,
                               model='damo/cv_unet_person-image-cartoon-handdrawn_compound-models')
    elif style == "sketch":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,
                               model='damo/cv_unet_person-image-cartoon-sketch_compound-models')
    elif style == "artstyle":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,
                               model='damo/cv_unet_person-image-cartoon-artstyle_compound-models')
    else:
        print('no such style %s'% style)
        return 0


    result = img_cartoon('input.png')

    cv2.imwrite('result_%s.png'%style, result[OutputKeys.OUTPUT_IMG])
    print('finished!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='anime')
    args = parser.parse_args()

    process(args)
