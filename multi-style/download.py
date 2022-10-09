from modelscope.hub.snapshot_download import snapshot_download
import argparse



def process(args):
    style = args.style
    print('download %s model'%style)
    if style == "anime":
        model_dir = snapshot_download('damo/cv_unet_person-image-cartoon_compound-models', cache_dir='.')

    elif style == "3d":
        model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-3d_compound-models', cache_dir='.')

    elif style == "handdrawn":
        model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-handdrawn_compound-models', cache_dir='.')

    elif style == "sketch":
        model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-sketch_compound-models', cache_dir='.')

    elif style == "artstyle":
        model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-artstyle_compound-models', cache_dir='.')

    else:
        print('no such style %s'% style)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='anime')
    args = parser.parse_args()

    process(args)
