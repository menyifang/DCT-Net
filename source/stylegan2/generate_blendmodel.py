import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import json
import glob

from PIL import Image

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def generate(args, g_ema, device, mean_latent, model_name, g_ema_ffhq):

    outdir = args.save_dir

    # print(outdir)
    # outdir = os.path.join(args.output, args.name, 'eval','toons_paired_0512')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            res, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            if args.form == "pair":
                sample_face, _ = g_ema_ffhq(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent
                )
                res = torch.cat([sample_face, res], 3)

            outpath = os.path.join(outdir, str(i).zfill(6)+'.png')
            utils.save_image(
                res,
                outpath,
                # f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            # print('save %s'% outpath)




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument('--config', type=str, default='config/conf_server_test_blend_shell.json')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')

    parser.add_argument('--form', type=str, default='single')
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()
    # from config updata paras
    opt = vars(args)
    with open(args.config) as f:
        config = json.load(f)['parameters']
        for key, value in config.items():
            opt[key] = value

    # args.ckpt = 'face_generation/experiment_stylegan/'+args.name+'/models_blend/G_blend_001000_4.pt'
    args.ckpt = 'face_generation/experiment_stylegan/'+args.name+'/models_blend/G_blend_'
    args.ckpt = glob.glob(args.ckpt+'*')[0]

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    # g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    ## add G_ffhq
    g_ema_ffhq = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint_ffhq = torch.load(args.ffhq_ckpt)
    g_ema_ffhq.load_state_dict(checkpoint_ffhq["g_ema"], strict=False)


    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    model_name = os.path.basename(args.ckpt)
    print('save generated samples to %s'% os.path.join(args.output, args.name, 'eval_blend', model_name))
    generate(args, g_ema, device, mean_latent, model_name, g_ema_ffhq)
    # generate_style_mix(args, g_ema, device, mean_latent, model_name, g_ema_ffhq)

    # latent_path = 'test2.pt'
    # generate_from_latent(args, g_ema, device, mean_latent, latent_path)
