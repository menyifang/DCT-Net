from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import os, cv2
import argparse

def load_cele_txt(celeb_file='celeb.txt'):
    celeb = open(celeb_file, 'r')
    lines = celeb.readlines()
    name_list = []
    for line in lines:
        name = line.strip('\n')
        if name != '':
            name_list.append(name)
    return name_list


def main(args):
    style = args.style
    repeat_num = 5

    model_id = 'damo/cv_cartoon_stable_diffusion_' + style
    pipe = pipeline(Tasks.text_to_image_synthesis, model=model_id,
                    model_revision='v1.0.0', torch_dtype=torch.float16)
    from diffusers.schedulers import EulerAncestralDiscreteScheduler
    pipe.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipeline.scheduler.config)
    print('model init finished!')


    save_dir = 'res_style_%s/syn_celeb' % (style)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = load_cele_txt('celeb.txt')
    person_num = len(name_list)
    for i in range(person_num):
        name = name_list[i]
        print('process %s' % name)

        if style == "clipart":
            prompt = 'archer style, a portrait painting of %s' % (name)
        else:
            prompt = 'sks style, a painting of a %s, no text' % (name)

        images = pipe({'text': prompt, 'num_images_per_prompt': repeat_num})['output_imgs']
        idx = 0
        for image in images:
            outpath = os.path.join(save_dir, '%s_%d.png' % (name, idx))
            cv2.imwrite(outpath, image)
            idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='clipart')

    args = parser.parse_args()
    main(args)

