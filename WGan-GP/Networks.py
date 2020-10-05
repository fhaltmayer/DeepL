import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, nf = 64, nz = 100, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, nf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d( nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d( nf *2, 3, 4, 2, 1, bias=False),
            nn.Tanh())      
        
    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, nf = 64, nz = 100, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),)

    def forward(self, input):
        return self.main(input)
