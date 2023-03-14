import cv2
import os
import numpy as np
import argparse
from source.facelib.facer import FaceAna
import source.utils as utils
from source.mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points
from modelscope.hub.snapshot_download import snapshot_download

class FaceProcesser:
    def __init__(self, dataroot, crop_size = 256, max_face = 1):
        self.max_face = max_face
        self.crop_size = crop_size
        self.facer = FaceAna(dataroot)

    def filter_face(self, lm, crop_size):
        a = max(lm[:, 0])-min(lm[:, 0])
        b = max(lm[:, 1])-min(lm[:, 1])
        # print("a:%d, b:%d"%(a,b))
        if max(a, b)<int(crop_size*0.3): # 眼间距 ？ 70
            return 0
        else:
            return 1

    def process(self, img):

        warped_face = None
        h, w, c = img.shape
        if c==4:
            img_bgr = img[:,:,:3]
        else:
            img_bgr = img

        src_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, landmarks, _ = self.facer.run(src_img)


        if boxes.shape[0] == 0:
            print('No face detected!')
            return warped_face

        # process all faces
        warped_faces = []
        i = 0

        for landmark in landmarks:
            if self.max_face and i>0:
                continue

            if self.filter_face(landmark, self.crop_size)==0:
                print("filtered!")
                continue

            f5p = utils.get_f5p(landmark, img_bgr)
            # face alignment
            warped_face, _ = warp_and_crop_face(
                img_bgr,
                f5p,
                ratio=0.75,
                reference_pts=get_reference_facial_points(default_square=True),
                crop_size=(self.crop_size, self.crop_size),
                return_trans_inv=True)

            warped_faces.append(warped_face)
            i = i+1


        return warped_faces




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="process remove bg result")
    parser.add_argument("--src_dir", type=str, default='', help="Path to src images.")
    parser.add_argument("--save_dir", type=str, default='', help="Path to save images.")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--max_face", type=int, default=1)
    parser.add_argument("--overwrite", type=int, default=1)
    args = parser.parse_args()
    args.save_dir = os.path.dirname(args.src_dir) + '/face_cartoon/raw_style_faces'

    crop_size = args.crop_size
    max_face = args.max_face
    overwrite = args.overwrite

    # model_dir = snapshot_download('damo/cv_unet_person-image-cartoon_compound-models', cache_dir='.')
    # print('model assets saved to %s'%model_dir)
    model_dir = 'damo/cv_unet_person-image-cartoon_compound-models'

    processer = FaceProcesser(dataroot=model_dir,crop_size=crop_size, max_face =max_face)

    src_dir = args.src_dir
    save_dir = args.save_dir

    # print('Step: start to extract aligned faces ... ...')

    print('src_dir:%s'% src_dir)
    print('save_dir:%s'% save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    paths = utils.all_file(src_dir)
    print('to process %d images'% len(paths))

    for path in sorted(paths):
        dirname = path[len(src_dir)+1:].split('/')[0]

        outpath = save_dir + path[len(src_dir):]
        if not overwrite:
            if os.path.exists(outpath):
                continue

        sub_dir = os.path.dirname(outpath)
        # print(sub_dir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        imgb = None
        imgc = None
        img = cv2.imread(path, -1)
        if img is None:
            continue

        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # print(img.shape)
        h,w,c = img.shape
        if h<256 or w<256:
            continue
        imgs = []

        # if need resize, resize here
        img_h, img_w, _ = img.shape
        warped_faces = processer.process(img)
        if warped_faces is None:
            continue
            # ### only for anime faces, single, not detect face
            # warped_face = imga

        i=0
        for res in warped_faces:
            # filter small faces
            h, w, c = res.shape
            if h < 256 or w < 256:
                continue
            outpath = os.path.join(os.path.dirname(outpath), os.path.basename(outpath)[:-4] + '_' + str(i) + '.png')

            cv2.imwrite(outpath, res)
            print('save %s' % outpath)
            i = i+1







