import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from pathlib import Path
from PIL import Image

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
    def __init__(self, pretrained=False):
        super(AlexNet, self).__init__()        
        self.F = nn.Sequential(*list(torchvision.models.alexnet(pretrained=pretrained).features))
        self.Pool = nn.AdaptiveAvgPool2d((6,6))
        self.C = nn.Sequential(*list(torchvision.models.alexnet(pretrained=pretrained).classifier[:-1]))
    def forward(self, x):
        x = self.F(x)
        x = self.Pool(x)
        x = torch.flatten(x, 1)
        x = self.C(x)
        return x

class HashNet(nn.Module):
    def __init__(self):
        super().__init__()
        NB_CLS=21
        N_bits = 64

        Baseline = AlexNet(False)
        fc_dim = 4096
        H = Hash_func(fc_dim, N_bits, NB_CLS)
        self.net = nn.Sequential(Baseline, H)

    def forward(self, x):
        return self.net(x)

_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_image(img:Image):
    return _image_transforms(img.convert("RGB")).unsqueeze(0)