from __future__ import print_function
import sys
import argparse
import gc
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
from copy import deepcopy
import glob
import re
from torchvision.utils import save_image


# Manual Seed is a mistake but I accidentally left it on for my training
# So certain seeds are way better trained and colored for the example since
# spot instances were shut down and rerun causing a reloop of the same seeds

# manualSeed = 999
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

# These are the directories where you have your data and want to store it.
# dataroot: location of image folder
# savedirectory: model checkpointing location
# log_directory: directory where the training log is placed
# img_directory: training image progress location

dataroot = "/media/fico/Data/Celeba/CelebAMask-HQ"

save_directory = "./Training/Saved_Models/"

log_directory = "./Training/"

img_directory = "./Training/Saved_Imgs/"

img_gen_directory = "./Training/End_Gen/"

# workers: How many threads for data loading
# nz: the size of the latent space: the larger the less feature loss theoretically
# lr: learning rate
# lambda_gp: lambda value for wasserstein loss gradient penalty
# img_batch_size: (image_res, batch_size) for each resolution of image
# betas: decay rates for adam
# steps: how many images per epoch, two epochs per resolution
# save_count: amount of steps to take before each checkpoint
# small_penalty_e: epsilon for fourth added loss value
workers = 4

nz = 512

lr = 0.001

lambda_gp = 10

img_batch_size = [(4,16),(8,16),(16,16),(32,16),(64,16),(128,16), (256, 14), (512, 6), (1024, 3)]

betas = (0, 0.99)

steps = 800000

save_count = 100000

small_penalty_e = .001

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# ref: https://github.com/hukkelas/progan-pytorch/blob/master/src/models/custom_layers.py
# Pixel norm takes the norm across every feature map, equation given in paper

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        div = torch.square(x)
        div = torch.mean(div, dim = 1, keepdim = True)
        div = div + 10**(-8)
        div = torch.square(div)
        return x/div

# Minibatch standard deviation takes the mean of the featuremaps across the minibatch and calculates the standard deviaton
# Small vlue of 10**(-8) to avoid division by zero. Standard deviation is repeated and attached as another channel to the featuremap.
class MiniBatchSTD(nn.Module):
    def __init__(self):
        super(MiniBatchSTD, self).__init__()

    def forward(self, x):
        s = x.shape
        std = x
        std = std - torch.mean(std, dim=0, keepdim= True)
        std = torch.mean(torch.square(std), dim=0)
        std = torch.sqrt(std + 10**(-8))
        std = torch.mean(std)
        std = std.to(x.dtype)
        std = std.repeat([s[0], 1, s[2], s[3]])
        std = torch.cat([x, std], 1)
        return std

# ref: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py
# This layer is a convolution layer where the weights are scaled at run time depending on
# how many parameteres there are as an input to the convolution. It uses he's weight initialization. 
class conv2d_e(nn.Module):
    def __init__(self, input_c, output_c, kernel, stride, pad):
        super(conv2d_e, self).__init__()
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(output_c, input_c, kernel, kernel)))
        self.bias = torch.nn.Parameter(torch.FloatTensor(output_c).fill_(0))
        self.stride = stride
        self.pad = pad
        fan_in = (kernel*kernel) * input_c
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return nn.functional.conv2d(input = x, 
                         weight = self.weight * self.scale, 
                         stride = self.stride, 
                         bias = self.bias, 
                         padding = self.pad)
        
# This layer is a linear layer where the weights are scaled at run time depending on
# how many parameteres there are as an input to the convolution. It uses he's weight initialization. 
class linear_e(nn.Module):
    def __init__(self, input_c, output_c):
        super(linear_e, self).__init__()
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(output_c, input_c)))
        self.bias = torch.nn.Parameter(torch.FloatTensor(output_c).fill_(0))
        fan_in = input_c
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return nn.functional.linear(input = x, 
                         weight = self.weight * self.scale, 
                         bias = self.bias)


# The generator structure, this code can be reduced to a Modulelist but that becomes very hard to debug
# due to logic of pulling certain layers for certain steps. For now, it is kept as a collection of if 
# statement blocks, each with the commented resolution above each block.

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.up_samp = nn.Upsample(scale_factor = 2)

        # 4
        self.start = linear_e(512, 8192)
        self.block = nn.Sequential(
            conv2d_e(nz, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end = conv2d_e(512, 3, 1, 1, 0)
        
        # 8
        self.block1 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),) 
        self.end1 = conv2d_e(512, 3, 1, 1, 0)
        
        # 16
        self.block2 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end2 = conv2d_e(512, 3, 1, 1, 0)
        
        # 32
        self.block3 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end3 = conv2d_e(512, 3, 1, 1, 0)
        
        # 64
        self.block4 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(256, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end4 = conv2d_e(256, 3, 1, 1, 0)
        
         # 128
        self.block5 = nn.Sequential(
            self.up_samp,
            conv2d_e(256, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(128, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end5 = conv2d_e(128, 3, 1, 1, 0)
        
         # 256
        self.block6 = nn.Sequential(
            self.up_samp,
            conv2d_e(128, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(64, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end6 = conv2d_e(64, 3, 1, 1, 0)

         # 512
        self.block7 = nn.Sequential(
            self.up_samp,
            conv2d_e(64, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(32, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end7 = conv2d_e(32, 3, 1, 1, 0)

         # 1024
        self.block8 = nn.Sequential(
            self.up_samp,
            conv2d_e(32, 16, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(16, 16, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end8 = conv2d_e(16, 3, 1, 1, 0)

    def forward(self, input, res, alpha):
        
        input1 = self.start(input)
        input1 = input1.view(-1,512,4,4)

        if res == 4:
            output = self.block(input1)
            output = self.end(output)

        elif res == 8:
            output = self.block(input1)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end(output_old)

            output = self.block1(output)
            output = self.end1(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old

        elif res == 16:
            output = self.block(input1)
            output = self.block1(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end1(output_old)

            output = self.block2(output)
            output = self.end2(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old

        elif res == 32:
            output = self.block(input1)
            output = self.block1(output)
            output = self.block2(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end2(output_old)

            output = self.block3(output)
            output = self.end3(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old

        elif res == 64:
            output = self.block(input1)
            output = self.block1(output)
            output = self.block2(output)
            output = self.block3(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end3(output_old)

            output = self.block4(output)
            output = self.end4(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old

        elif res == 128:
            output = self.block(input1)
            output = self.block1(output)
            output = self.block2(output)
            output = self.block3(output)
            output = self.block4(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end4(output_old)

            output = self.block5(output)
            output = self.end5(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old

        elif res == 256:
            output = self.block(input1)
            output = self.block1(output)
            output = self.block2(output)
            output = self.block3(output)
            output = self.block4(output)
            output = self.block5(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end5(output_old)

            output = self.block6(output)
            output = self.end6(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old

        elif res == 512:
            output = self.block(input1)
            output = self.block1(output)
            output = self.block2(output)
            output = self.block3(output)
            output = self.block4(output)
            output = self.block5(output)
            output = self.block6(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end6(output_old)

            output = self.block7(output)
            output = self.end7(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old
            
        elif res == 1024:
            output = self.block(input1)
            output = self.block1(output)
            output = self.block2(output)
            output = self.block3(output)
            output = self.block4(output)
            output = self.block5(output)
            output = self.block6(output)
            output = self.block7(output)
            if alpha >= 0:
                output_old = self.up_samp(output)
                output_old = self.end7(output_old)

            output = self.block8(output)
            output = self.end8(output)
            if alpha >= 0:
                output = alpha*output + (1-alpha)*output_old
            
        return output

# The discriminator structure, this code can be reduced to a Modulelist but that becomes very hard to debug
# due to logic of pulling certain layers for certain steps. For now, it is kept as a collection of if 
# statement blocks, each with the commented resolution above each block.

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down_samp = nn.AvgPool2d(2)
    
        # 4
        self.block = nn.Sequential(
            MiniBatchSTD(),
            conv2d_e(513, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 4, 1, 0),
            nn.LeakyReLU(.2),
            nn.Flatten(),
            linear_e(512, 1))
        self.start = conv2d_e(3, 512, 1, 1, 0)
        
        # 8
        self.block1 = nn.Sequential(
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start1 = conv2d_e(3, 512, 1, 1, 0)

        # 16
        self.block2 = nn.Sequential(
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start2 = conv2d_e(3, 512, 1, 1, 0)
        
        # 32
        self.block3 = nn.Sequential(
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start3 = conv2d_e(3, 512, 1, 1, 0)
        
        # 64
        self.block4 = nn.Sequential(
            conv2d_e(256, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(256, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start4 = conv2d_e(3, 256, 1, 1, 0)
        
        # 128
        self.block5= nn.Sequential(
            conv2d_e(128, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(128, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start5 = conv2d_e(3, 128, 1, 1, 0)

        # 256
        self.block6= nn.Sequential(
            conv2d_e(64, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(64, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start6 = conv2d_e(3, 64, 1, 1, 0)

        # 512
        self.block7= nn.Sequential(
            conv2d_e(32, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(32, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start7 = conv2d_e(3, 32, 1, 1, 0)

        # 1024
        self.block8= nn.Sequential(
            conv2d_e(16, 16, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(16, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start8 = conv2d_e(3, 16, 1, 1, 0)





    
    def forward(self, input, res, alpha):   
        
        if res == 4:
            output = self.start(input)
            output = self.block(output)
            
        elif res == 8:
            output = self.start1(input)
            output = self.block1(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start(output_old)
                output = alpha*output + (1-alpha)*output_old
            
            output = self.block(output)
        
        elif res == 16:
            output = self.start2(input)
            output = self.block2(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start1(output_old)
                output = alpha*output + (1-alpha)*output_old
                
            output = self.block1(output)
            output = self.block(output)
            
        elif res == 32:
            output = self.start3(input)
            output = self.block3(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start2(output_old)
                output = alpha*output + (1-alpha)*output_old
            
            output = self.block2(output)
            output = self.block1(output)
            output = self.block(output)
            
        elif res == 64:
            output = self.start4(input)
            output = self.block4(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start3(output_old)
                output = alpha*output + (1-alpha)*output_old
                
            output = self.block3(output)
            output = self.block2(output)
            output = self.block1(output)
            output = self.block(output)
            
        elif res == 128:
            output = self.start5(input)
            output = self.block5(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start4(output_old)
                output = alpha*output + (1-alpha)*output_old
            
            output = self.block4(output)
            output = self.block3(output)
            output = self.block2(output)
            output = self.block1(output)
            output = self.block(output)
        
        elif res == 256:
            output = self.start6(input)
            output = self.block6(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start5(output_old)
                output = alpha*output + (1-alpha)*output_old
            
            output = self.block5(output)
            output = self.block4(output)
            output = self.block3(output)
            output = self.block2(output)
            output = self.block1(output)
            output = self.block(output)
        
        elif res == 512:
            output = self.start7(input)
            output = self.block7(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start6(output_old)
                output = alpha*output + (1-alpha)*output_old
                         
            output = self.block6(output)
            output = self.block5(output)
            output = self.block4(output)
            output = self.block3(output)
            output = self.block2(output)
            output = self.block1(output)
            output = self.block(output)
        
        elif res == 1024:
            output = self.start8(input)
            output = self.block8(output)
            if alpha >= 0:
                output_old = self.down_samp(input)
                output_old = self.start7(output_old)
                output = alpha*output + (1-alpha)*output_old
            
            output = self.block7(output)             
            output = self.block6(output)
            output = self.block5(output)
            output = self.block4(output)
            output = self.block3(output)
            output = self.block2(output)
            output = self.block1(output)
            output = self.block(output)
            
        return output

# Creates the data loaders for each resolution before training
def load_data():
    data_loaders = []
    for img_size, batch_size in img_batch_size:
        dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

        dataload = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers, drop_last=True)
        data_loaders.append(dataload)
    # print("Data Loaded")
    return data_loaders

# ref: https://discuss.pytorch.org/t/copy-weights-only-from-a-networks-parameters/5841
# Expenential running average for the geneartor, this is used to smooth out the 
# weights across all the training when generating images, mentioned in paper
# and really helps with avoiding shading probelems that randomly occur when ending on a training minibatch
def update_running_avg(original, copy):
    with torch.no_grad(): 
        params1 = original.named_parameters()
        params2 = copy.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_((1 - .999) * dict_params2[name1] + (.999) * param1.data)

# Loading up the latest saved model if load_train is true, if not it sets the values to the default
# starting values.
def startup(load_train, mixed_precision):
    def extract_number(f):
        s = re.findall("\d+",f)
        # print(''.join(s))
        return (int(''.join(s)) if s else -1,f)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG_copy = deepcopy(netG)
    
    scalerD = -1
    scalerG = -1
    
    fixed_noise = torch.randn(16, nz, device=device)
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)
    
    training = True
    current_data = 0
    res = 4
    fade_in = False
    epoch = 0
    count = 0
    
    if load_train:
        if not mixed_precision:
            pathD = save_directory + "Regular/" + "D/"
            pathG = save_directory + "Regular/" + "G/"

        else:
            pathD = save_directory + "Amp/" + "D/"
            pathG = save_directory + "Amp/" + "G/"

        try:  
            list_of_files = glob.glob(pathD + '*')
            latest_file = max(list_of_files,key=extract_number)
            checkD = torch.load(latest_file)

            list_of_files = glob.glob(pathG + '*')
            latest_file = max(list_of_files,key=extract_number)
            checkG = torch.load(latest_file)

            optimizerD.load_state_dict(checkD['optimizer_state_dict'])
            netD.load_state_dict(checkD['model_state_dict'])
            optimizerG.load_state_dict(checkG['optimizer_state_dict'])
            netG.load_state_dict(checkG['model_state_dict'])
            netG_copy.load_state_dict(checkG['copy_model_state_dict'])
            if not mixed_precision:
                scalerD = -1
                scalerG = -1

            else:
                scalerD = torch.cuda.amp.GradScaler()
                scalerG = torch.cuda.amp.GradScaler()
                scalerD.load_state_dict(checkD['scaler_state_dict'])
                scalerG.load_state_dict(checkG['scaler_state_dict'])

            training = checkD['training']
            epoch = checkD['next_epoch']
            res = checkD['next_res']
            current_data = checkD['next_dict']
            fade_in = checkD['next_fade']
            fixed_noise = checkG['fixednoise']
            count = checkD['count']
            with open(log_directory + "log.txt","a+") as f:
                print(file=f)
                print("Loaded with epoch: " + str(epoch) + " and count: " + str(count), file=f)

        except:
            with open(log_directory + "log.txt","a+") as f:
                print(file=f)
                print("Starting new with epoch: " + str(epoch) + " and count: " + str(count), file=f)

    return [netD, netG, netG_copy, optimizerD, optimizerG, scalerD, scalerG, epoch, res, current_data, fade_in, fixed_noise, training, count]

# Calculations for the gradient penalty taken from wgan-gp. Takes a random interpolation between the generated batch and real image batch and
# calculates the gradient for that image. Then the norm is taken of that gradient. 
def gradient_penalty(netD, netG, mini_batch, real_imgs, fake_imgs, alpha, res, mixed_precision):
    gp_alpha = torch.randn(mini_batch, 1, 1, 1, device = device)
    interp = gp_alpha * real_imgs + ((1-gp_alpha) * fake_imgs.detach())
    interp.requires_grad = True      
    if mixed_precision:
        pass

    else:
        model_interp = netD(interp, alpha = alpha, res = res)

    if mixed_precision:
        pass
    
    else:
        grads = torch.autograd.grad(outputs=model_interp, inputs=interp,
                      grad_outputs=torch.ones(model_interp.size()).to(device),
                      create_graph=True, retain_graph=True, only_inputs=True)[0]

        grads = torch.square(grads)
        grads = torch.sum(grads, dim = [1,2,3])
        grads = torch.sqrt(grads)
        grads = grads - 1
        grads = torch.square(grads)
        grad_pen = grads * lambda_gp
        return grad_pen

# Small penalty attached onto wgan-gp loss specified in paper
def small_penalty(netD, output_real, mixed_precision = False):
    if mixed_precision:
        pass
    else:
        penalty = torch.square(output_real)
        penalty = penalty * small_penalty_e
    return penalty

# Logging function that prints training log into log.txt and to the console. It also generates 16 samples
# at the current step and saves them.
def logging(epoch, res, fade_in, count, alpha, loss_D, loss_G, netG_copy, fixed_noise, mixed_precision):
    with open(log_directory + "log.txt","a+") as f:
        print("Res:", res, "Fade_in:", fade_in, "Iter:",count, "alpha:", alpha, file=f)
        print("W with GP:", loss_D.item(),  "Loss G:", loss_G.item(), file=f)
    print("Res:", res, "Fade_in:", fade_in, "Iter:",count, "alpha:", alpha)
    print("W with GP:", loss_D.item(),  "Loss G:", loss_G.item())
    print()
    
    with torch.no_grad():
        guess = netG_copy(fixed_noise, res = res, alpha=alpha)
        guess = guess.cpu()
        
        old_min = torch.min(guess)
        old_max = torch.max(guess)
        old_range = old_max - old_min
        new_range = 1 - 0
       
        guess = (((guess - old_min)*new_range)/ old_range) + 0
        guess = guess.permute(0,2,3,1)

        fig = plt.figure(figsize=(4,4))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(guess[i, :, :])
            plt.axis('off')
        if mixed_precision:
            path = img_directory + "Training_Imgs_AMP/Res:" + str(res) + "_fade_in:" + str(fade_in) + "_training_step:" + str(count) + ".png"
        else:
            path = img_directory + "Training_Imgs/Res:" + str(res) + "_fade_in:" + str(fade_in) + "_training_step:" + str(count) + ".png"
        plt.savefig(path, dpi=300)
        plt.close('all')

# Saving the model and the current point in training for easy resume if there is a crash in training
def save_models(count, epoch, res, current_data, fade_in, netD, optimizerD, netG, netG_copy, optimizerG, fixed_noise, scalerD, scalerG, training, mixed_precision):
    if fade_in == True:
        tempF = 0

    else:
        tempF = 1

    if not mixed_precision:
        pathD = save_directory + "Regular/" + "D/" + "next_res:" + str(res) + "next_fade:" + str(tempF)
        pathG = save_directory + "Regular/" + "G/" + "next_res:" + str(res) + "next_fade:" + str(tempF)
        torch.save({
            'training': training,
            'next_epoch': epoch,
            'next_res': res,
            'next_dict': current_data,
            'next_fade': fade_in,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'count': count,
            }, pathD)

        torch.save({
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'fixednoise': fixed_noise,
            'copy_model_state_dict': netG_copy.state_dict(),
            }, pathG)
            
    else:
        pathD = save_directory + "Amp/" + "D/" + "next_res:" + str(res) + "next_fade:" + str(tempF)
        pathG = save_directory + "Amp/" + "G/" + "next_res:" + str(res) + "next_fade:" + str(tempF) 
        torch.save({
            'training': training,
            'next_epoch': epoch,
            'next_res': res,
            'next_dict': current_data,
            'next_fade': fade_in,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'scaler_state_dict': scalerD.state_dict(),
            'count': count,
            }, pathD)

        torch.save({
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'fixednoise': fixed_noise,
            'scaler_state_dict': scalerG.state_dict()
            }, pathG)

    with open(log_directory + "log.txt","a+") as f:
        print("Saved with epoch: " + str(epoch) + " and count: " + str(count), file=f)

# Creating all the necessary directories if they do not exist already
def check_directories():
    if not os.path.exists(save_directory + "Regular/D/"):
            os.makedirs(save_directory + "Regular/D/")
    if not os.path.exists(save_directory + "Regular/G/"):
            os.makedirs(save_directory + "Regular/G/")
    if not os.path.exists(save_directory + "Amp/D/"):
            os.makedirs(save_directory + "Amp/D/")
    if not os.path.exists(save_directory + "Amp/G/"):
            os.makedirs(save_directory + "Amp/G/")
    if not os.path.exists(img_directory + "Training_Imgs_AMP/"):
            os.makedirs(img_directory + "Training_Imgs_AMP/")
    if not os.path.exists(img_directory + "Training_Imgs/"):
            os.makedirs(img_directory + "Training_Imgs/")
    if not os.path.exists(img_gen_directory):
            os.makedirs(img_gen_directory)
    img_gen_directory

# Training loop, 2 epochs per resolution, first one epoch with fading in the new layer, then one without fade
def training(load_train = False, mixed_precision = False):
    with open(log_directory + "log.txt","a+") as f:
        values = startup(load_train, mixed_precision)
        netD = values[0]
        netG = values[1]
        netG_copy = values[2]
        optimizerD = values[3]
        optimizerG = values[4]
        scalerD = values[5]
        scalerG = values[6]
        epoch = values[7]
        res = values[8]
        current_data = values[9]
        fade_in = values[10]
        fixed_noise = values[11]
        training = values[12]
        count = values[13]
        data_loaders = load_data()
        
    while(training):

        loader = iter(data_loaders[current_data])
     
        start_time = time.time()
        
        # Repeating the dataloader if it has been fully read through
        while count < steps:
            try:
                img = loader.next()

            except StopIteration:
                loader = iter(data_loaders[current_data])
                img = loader.next()

            # Calculating apha for the current step, -1 if no fading is done
            if not fade_in:
                alpha = -1
               
            else:
                alpha = count/steps
                
            mini_batch = len(img[0])
    
            real_imgs = img[0].to(device)
        
            noise = torch.randn(mini_batch, nz, device=device)
        
            netD.zero_grad()
            
            # Discriminator loss on real images
            if mixed_precision:
                pass
            else:
                output_real = netD(real_imgs, alpha=alpha, res=res).squeeze()

            
            # Discriminator loss on fake images
            if mixed_precision:
                pass   
            else:
                fake_imgs = netG(noise, res=res, alpha=alpha)
                output_fake = netD(fake_imgs.detach(), alpha=alpha, res=res).squeeze()
                
            # Gradient Penalty
            grad_pen = gradient_penalty(netD, netG, mini_batch, real_imgs, fake_imgs, alpha, res, mixed_precision)
            

            # Extra small penalty
            penalty = small_penalty(netD, output_real, mixed_precision)
                

            # Calculating entire loss and taking step
            if mixed_precision:
                pass
            else:
                loss_D = torch.mean(output_fake - output_real + grad_pen + penalty)
                loss_D.backward()
                optimizerD.step()      
                
            netG.zero_grad()
            
            # Generator loss on its fake batch
            if mixed_precision:
                pass
            else:
                output = netD(fake_imgs, alpha=alpha, res=res).squeeze()
                loss_G = -torch.mean(output)
                loss_G.backward()
                optimizerG.step()
            
            # Update the running average of generator weights
            update_running_avg(netG, netG_copy)
                
            count += mini_batch

            # Log every 5000 steps
            if count %5000 < mini_batch:
                logging(epoch, res, fade_in, count, alpha, loss_D, loss_G, netG_copy, fixed_noise, mixed_precision)

            # Save model every save_count, if the resolution is above 64, log even more often due to large slowdown in speed
            if res <= 64:
                if count%save_count < mini_batch:
                    save_models(count, epoch, res, current_data, fade_in, netD, optimizerD, netG, netG_copy, optimizerG, fixed_noise, scalerD, scalerG, training, mixed_precision)

            else:
                if count%(save_count//4) < mini_batch:
                    save_models(count, epoch, res, current_data, fade_in, netD, optimizerD, netG, netG_copy, optimizerG, fixed_noise, scalerD, scalerG, training, mixed_precision)

        # Update training loop resolution, and if currently fading or not fading in new layer
        if fade_in == False:
            fade_in = True
            current_data += 1
            if current_data == len(data_loaders):
                training= False

            res = res * 2

        else:
            fade_in = False

        epoch += 1 
        count = 0
        
        # Save model after every eoch, 2 epochs per resolution
        save_models(count, epoch, res, current_data, fade_in, netD, optimizerD, netG, netG_copy, optimizerG, fixed_noise, scalerD, scalerG, training, mixed_precision)

        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch time: ", epoch_time)

def generate_images(counts, mixed_precision = False):
    def extract_number(f):
        s = re.findall("\d+",f)
        # print(''.join(s))
        return (int(''.join(s)) if s else -1,f)
    
    if not mixed_precision:
        pathD = save_directory + "Regular/" + "D/"
        pathG = save_directory + "Regular/" + "G/"

    else:
        pathD = save_directory + "Amp/" + "D/"
        pathG = save_directory + "Amp/" + "G/"

     
    list_of_files = glob.glob(pathD + '*')
    latest_file = max(list_of_files,key=extract_number)
    checkD = torch.load(latest_file)

    list_of_files = glob.glob(pathG + '*')
    latest_file = max(list_of_files,key=extract_number)
    checkG = torch.load(latest_file)

    netG_copy = Generator().to(device)

    training = checkD['training']
    epoch = checkD['next_epoch']
    res = checkD['next_res']
    current_data = checkD['next_dict']
    fade_in = checkD['next_fade']
    fixed_noise = checkG['fixednoise']
    count = checkD['count']
    netG_copy.load_state_dict(checkG['copy_model_state_dict'])

    if count == 0 and fade_in == True:
        res = res/2

    with torch.no_grad():
        noise = torch.randn(counts, nz, device=device)
        output = netG_copy(noise, res = res, alpha = -1)
        print(output.shape)        
        old_min = torch.min(output)
        old_max = torch.max(output)
        old_range = old_max - old_min
        new_range = 1 - 0
       
        output = (((output - old_min)*new_range)/ old_range) + 0

    for x in range(counts):
        img = output[x]

        save_image(img, img_gen_directory + str(x) + ".png" )

def generate_video(mixed_precision = False):
    def extract_number(f):
        s = re.findall("\d+",f)
        return (int(''.join(s)) if s else -1,f)

    if mixed_precision:
        path = img_directory + "Training_Imgs_AMP/"

    else:
        path = img_directory + "Training_Imgs/"

    for file in os.listdir(path):
        if file.endswith(".png"):
            if file.find("False") > 0:
                print("False")
                os.rename(os.path.join(path, file), os.path.join(path, file.replace("False", "1")))
            elif file.find("True") > 0:
                os.rename(os.path.join(path, file), os.path.join(path, file.replace("True", "0")))
    
    for filename in os.listdir(path):
        prefix, num = filename[:-4].split('step:')
        num = num.zfill(6)
        new_filename = prefix + "step:" + num + ".png"
        os.rename(os.path.join(path , filename), os.path.join(path , new_filename))
    
    size = (1200,1200)
    list_of_files = glob.glob(path + '*.png')
    list_of_files.sort(key=extract_number)
    out = cv2.VideoWriter('Timelapse.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for filename in list_of_files:
        img = cv2.imread(filename)
        out.write(img)
    out.release()
    


# arg > 0, test if there is gpu space to train each resolution
# arg = 0: train the network
# arg = -1: generate images with latest network
# arg = -2: generate training video
def main(argv):
    arg = int(argv[0])
    if arg > 0:
        size = 16
        if arg == 256:
            size = 14
        elif arg == 512:
            size = 6
        elif arg == 1024:
            size = 3
        print(argv[0])
        fixed_noise = torch.randn(16, nz, device=device)
        noise = torch.randn(size, nz, device=device)
        netG = Generator().to(device)
        netD = Discriminator().to(device)
        netG_copy = deepcopy(netG)
        test  = netG(noise, alpha = .5, res = arg)
        test1 = netD(test, alpha = .5, res = arg)
        update_running_avg(netG, netG_copy)
        with torch.no_grad():
            test2 = netG_copy(fixed_noise, alpha = .5, res = arg)
        print("success")
        return 0
    
    elif arg == 0:
        check_directories()
        training(load_train = True, mixed_precision=False)
    
    elif arg == -1:
        check_directories()
        generate_images(50)
    
    elif arg == -2:
        generate_video()

if __name__ == "__main__":
    main(sys.argv[1:])