import torch
from torch import nn
from .vgg import VGG
import os
from configs.paths_config import model_paths

class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()
        print("Loading VGG19 model from path: {}".format(model_paths["vgg"]))

        self.vgg_model = VGG()
        self.vgg_model.load_state_dict(torch.load(model_paths['vgg']))
        self.vgg_model.cuda()
        self.vgg_model.eval()

        self.l1loss = torch.nn.L1Loss()





    def forward(self, input_photo, output):
        vgg_photo = self.vgg_model(input_photo)
        vgg_output = self.vgg_model(output)
        n, c, h, w = vgg_photo.shape
        # h, w, c = vgg_photo.get_shape().as_list()[1:]
        loss = self.l1loss(vgg_photo, vgg_output)

        return loss
