import os
import cv2
from modelscope.trainers.cv import CartoonTranslationTrainer


def main(args):

    data_photo = os.path.join(args.data_dir, 'face_photo')
    data_cartoon = os.path.join(args.data_dir, 'face_cartoon')

    style = args.style
    if style == "anime":
        style = ""
    else:
        style = '-' + style
    model_id = 'damo/cv_unet_person-image-cartoon' + style + '_compound-models'

    max_steps = 300000
    trainer = CartoonTranslationTrainer(
        model=model_id,
        work_dir=args.work_dir,
        photo=data_photo,
        cartoon=data_cartoon,
        max_steps=max_steps)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process remove bg result")
    parser.add_argument("--data_dir", type=str, default='', help="Path to training images.")
    parser.add_argument("--work_dir", type=str, default='', help="Path to save results.")
    parser.add_argument("--style", type=str, default='anime', help="resume training from similar style.")

    args = parser.parse_args()

    main(args)

