# model blending technique
import os
import cv2 as cv
import torch
from model import Generator
import math
import argparse

def extract_conv_names(model):
    model = list(model.keys())
    conv_name = []
    resolutions = [4*2**x for x in range(9)]
    level_names = [["Conv0_up", "Const"], ["Conv1", "ToRGB"]]


def blend_models(model_1, model_2, resolution, level, blend_width=None):
    resolutions = [4 * 2 ** i for i in range(7)]
    mid = resolutions.index(resolution)

    device = "cuda"

    size = 256
    latent = 512
    n_mlp = 8
    channel_multiplier =2
    G_1 = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    ckpt_ffhq = torch.load(model_1, map_location=lambda storage, loc: storage)
    G_1.load_state_dict(ckpt_ffhq["g"], strict=False)


    G_2 = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    ckpt_toon = torch.load(model_2)
    G_2.load_state_dict(ckpt_toon["g_ema"])



    # G_1 = stylegan2.models.load(model_1)
    # G_2 = stylegan2.models.load(model_2)
    model_1_state_dict = G_1.state_dict()
    model_2_state_dict = G_2.state_dict()
    assert(model_1_state_dict.keys() == model_2_state_dict.keys())
    G_out = G_1.clone()

    layers = []
    ys = []
    for k, v in model_1_state_dict.items():
        if k.startswith('convs.'):
            pos = int(k[len('convs.')])
            x = pos - mid
            if blend_width:
                exponent = -x / blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 0 else 0

            layers.append(k)
            ys.append(y)
        elif k.startswith('to_rgbs.'):
            pos = int(k[len('to_rgbs.')])
            x = pos - mid
            if blend_width:
                exponent = -x / blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 0 else 0
            layers.append(k)
            ys.append(y)
    out_state = G_out.state_dict()
    for y, layer in zip(ys, layers):
        out_state[layer] = y * model_2_state_dict[layer] + \
            (1 - y) * model_1_state_dict[layer]
        print('blend layer %s'%str(y))
    G_out.load_state_dict(out_state)
    return G_out


def blend_models_2(model_1, model_2, resolution, level, blend_width=None):
    # resolution = f"{resolution}x{resolution}"
    resolutions = [4 * 2 ** i for i in range(7)]
    mid = [resolutions.index(r) for r in resolution]

    G_1 = stylegan2.models.load(model_1)
    G_2 = stylegan2.models.load(model_2)
    model_1_state_dict = G_1.state_dict()
    model_2_state_dict = G_2.state_dict()
    assert(model_1_state_dict.keys() == model_2_state_dict.keys())
    G_out = G_1.clone()

    layers = []
    ys = []
    for k, v in model_1_state_dict.items():
        if k.startswith('G_synthesis.conv_blocks.'):
            pos = int(k[len('G_synthesis.conv_blocks.')])
            y = 0 if pos in mid else 1
            layers.append(k)
            ys.append(y)
        elif k.startswith('G_synthesis.to_data_layers.'):
            pos = int(k[len('G_synthesis.to_data_layers.')])
            y = 0 if pos in mid else 1
            layers.append(k)
            ys.append(y)
    # print(ys, layers)
    out_state = G_out.state_dict()
    for y, layer in zip(ys, layers):
        out_state[layer] = y * model_2_state_dict[layer] + \
            (1 - y) * model_1_state_dict[layer]
    G_out.load_state_dict(out_state)
    return G_out


def main(name):

    resolution = 4

    model_name = '001000.pt'

    G_out = blend_models("pretrained_models/stylegan2-ffhq-config-f-256-550000.pt",
                         "face_generation/experiment_stylegan/"+name+"/models/"+model_name,
                         resolution,
                         None)
    # G_out.save('G_blend.pth')
    outdir = os.path.join('face_generation/experiment_stylegan',name,'models_blend')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'G_blend_'+str(model_name[:-3])+'_'+ str(resolution)+'.pt')
    torch.save(
        {
            "g_ema": G_out.state_dict(),
        },
        # 'G_blend_570000_16.pth',
        outpath
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="style blender")
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    print('model name:%s'%args.name)
    main(args.name)