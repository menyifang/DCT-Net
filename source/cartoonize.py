import os
import cv2
import tensorflow as tf
import numpy as np
from source.facelib.facer import FaceAna
import source.utils as utils
from source.mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()


class Cartoonizer():
    def __init__(self, dataroot):

        self.facer = FaceAna(dataroot)
        self.sess_head = self.load_sess(
            os.path.join(dataroot, 'cartoon_anime_h.pb'), 'model_head')
        self.sess_bg = self.load_sess(
            os.path.join(dataroot, 'cartoon_anime_bg.pb'), 'model_bg')

        self.box_width = 288
        global_mask = cv2.imread(os.path.join(dataroot, 'alpha.jpg'))
        global_mask = cv2.resize(
            global_mask, (self.box_width, self.box_width),
            interpolation=cv2.INTER_AREA)
        self.global_mask = cv2.cvtColor(
            global_mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def load_sess(self, model_path, name):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        print(f'loading model from {model_path}')
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name=name)
            sess.run(tf.global_variables_initializer())
        print(f'load model {model_path} done.')
        return sess


    def detect_face(self, img):
        src_h, src_w, _ = img.shape
        src_x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, landmarks, _ = self.facer.run(src_x)
        if boxes.shape[0] == 0:
            return None
        else:
            return landmarks


    def cartoonize(self, img):
        # img: RGB input
        ori_h, ori_w, _ = img.shape
        img = utils.resize_size(img, size=720)

        img_brg = img[:, :, ::-1]

        # background process
        pad_bg, pad_h, pad_w = utils.padTo16x(img_brg)

        bg_res = self.sess_bg.run(
            self.sess_bg.graph.get_tensor_by_name(
                'model_bg/output_image:0'),
            feed_dict={'model_bg/input_image:0': pad_bg})
        res = bg_res[:pad_h, :pad_w, :]

        landmarks = self.detect_face(img_brg)
        if landmarks is None:
            print('No face detected!')
            return res

        print('%d faces detected!'%len(landmarks))
        for landmark in landmarks:
            # get facial 5 points
            f5p = utils.get_f5p(landmark, img_brg)

            # face alignment
            head_img, trans_inv = warp_and_crop_face(
                img,
                f5p,
                ratio=0.75,
                reference_pts=get_reference_facial_points(default_square=True),
                crop_size=(self.box_width, self.box_width),
                return_trans_inv=True)

            # head process
            head_res = self.sess_head.run(
                self.sess_head.graph.get_tensor_by_name(
                    'model_head/output_image:0'),
                feed_dict={
                    'model_head/input_image:0': head_img[:, :, ::-1]
                })

            # merge head and background
            head_trans_inv = cv2.warpAffine(
                head_res,
                trans_inv, (np.size(img, 1), np.size(img, 0)),
                borderValue=(0, 0, 0))

            mask = self.global_mask
            mask_trans_inv = cv2.warpAffine(
                mask,
                trans_inv, (np.size(img, 1), np.size(img, 0)),
                borderValue=(0, 0, 0))
            mask_trans_inv = np.expand_dims(mask_trans_inv, 2)

            res = mask_trans_inv * head_trans_inv + (1 - mask_trans_inv) * res

        res = cv2.resize(res, (ori_w, ori_h), interpolation=cv2.INTER_AREA)

        return res




