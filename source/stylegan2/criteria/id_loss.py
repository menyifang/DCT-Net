import torch
from torch import nn
from .model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        model_paths = '/data/vdb/qingyao/cartoon/mycode/pretrained_models/model_ir_se50.pth'
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_input = y_hat_feats[i].dot(x_feats[i])
            id_logs.append({
                            'diff_input': float(diff_input)
                            })
            # loss += 1 - diff_target
            # modify
            loss += 1 - diff_input
            count += 1

        return loss / count
