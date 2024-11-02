import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from pathlib import Path
from PIL import Image
import timm

class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(torch.FloatTensor(NB_CLS, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        X = self.Hash(X)
        return torch.tanh(X)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()        
        self.F = nn.Sequential(*list(torchvision.models.alexnet(weights=None).features))
        self.Pool = nn.AdaptiveAvgPool2d((6,6))
        self.C = nn.Sequential(*list(torchvision.models.alexnet(weights=None).classifier[:-1]))
    def forward(self, x):
        x = self.F(x)
        x = self.Pool(x)
        x = torch.flatten(x, 1)
        x = self.C(x)
        return x
    def get_dim(self):
        return 4096

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pretrained = torchvision.models.resnet50(weights=None)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return x

    def get_dim(self):
        return 2048

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.pm = timm.create_model('vit_base_patch16_224', pretrained=False)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

    def get_dim(self):
        return 768

class DeiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.pm = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, self.pm.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

    def get_dim(self):
        return 768


class HashNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        NB_CLS=21
        N_bits = 64

        Baseline = backbone
        fc_dim = backbone.get_dim()
        H = Hash_func(fc_dim, N_bits, NB_CLS)
        self.net = nn.Sequential(Baseline, H)

    def forward(self, x):
        return self.net(x)

_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_image(img:Image):
    return _image_transforms(img.convert("RGB")).unsqueeze(0)