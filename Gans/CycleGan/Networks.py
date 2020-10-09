import torch
import torch.nn as nn
import random

class Residual2d(nn.Module):
    def __init__(self, dim):
        super(Residual2d, self).__init__()
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
            nn.InstanceNorm2d(dim))
        # self.relu = nn.ReLU(True)
    
    def forward(self, x):
        output = x + self.res(x)
        # output = self.relu(output)
        return output

        
# Generator Code
# Use mirrored padding always or just beginning and final layer
class Generator(nn.Module):
    def __init__(self, rb = 9):
        super(Generator, self).__init__()
        # 3x256x256 -> 64x256x256

        self.c7s1_64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 1, 0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True))
    
        # 64x256x256 -> 128x128x128
        self.d128 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2, 0),
            nn.InstanceNorm2d(128),
            nn.ReLU(True))
        
        # 128x128x128 -> 256x64x64
        self.d256 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3, 2, 0),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))
        
        # 256x64x64 -> 256x64x64
        res_blocks = [Residual2d(256) for x in range(rb)]
        self.R9 = nn.Sequential(*res_blocks)
        
        # Try using upsample + conv
        # For some reason padding beforehand doesnt work with convtranpose, 
        # should work even with tranpose special padding rule
        # 256x64x64 -> 128x128x128
        self.u128 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        
        # 128x128x128 -> 64x256x256
        self.u256 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.c7s1_3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0),
            # nn.InstanceNorm2d(3),
            # nn.ReLU(True),
            nn.Tanh())
            
        
    def forward(self, input):
        output = self.c7s1_64(input)
        output = self.d128(output)
        output = self.d256(output)
        output= self.R9(output)
        output= self.u128(output)
        output= self.u256(output)
        output= self.c7s1_3(output)
        # print(output.shape)
        return output




# https://arxiv.org/pdf/1611.07004.pdf
# Try different patch sizes
# Last two convolutions need to have a stride and padding of 1
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.C64 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace= True))
    
        self.C128 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace= True))
        
        self.C256 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace= True))
    
        self.C512 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace= True))
        self.last = nn.Conv2d(512,1, 4, 1, 1)
        
    def forward(self, input):
        output = self.C64(input)
        output = self.C128(output)
        output = self.C256(output)
        output = self.C512(output)
        output = self.last(output)



        # print(output.shape)
        return output

# Change buffering to each image in batch instead of entire batch
class ImageBuffer:
    def __init__(self):
        self.buffer = []
        self.buffer_size = 0    
    def grab(self, batch):
        returnals = []
        for x in batch:
            img = torch.unsqueeze(x, 0)
            # print(img.shape)
            if self.buffer_size < (50):
                self.buffer.append(img)
                self.buffer_size += 1
                returnals.append(img)
            else:
                t =random.uniform(0, 1)
                # print(t)
                if t > 0.5:
                    returnals.append(img)
                else:
                    # print("buffed mean: " + str(torch.mean(img)))
                    pos = random.randrange(0, self.buffer_size)
                    temp = self.buffer[pos].clone()
                    self.buffer[pos] = img
                    # print("selected mean: " + str(torch.mean(temp)))
                    returnals.append(temp)  
        z = torch.cat(returnals, 0)
        # print(z.shape)
        return z

