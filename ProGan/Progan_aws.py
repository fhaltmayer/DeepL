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
from apex import amp
import time
from copy import deepcopy
import glob
# import logging

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "/Data/Celeba/CelebAMask-HQ"

save_directory = "/Data/Training/Saved_Models/"

log_directory = "/Data/Training/"

img_directory = "/Data/Training/Saved_Imgs/"

workers = 4

nz = 512

lr = 0.001

beta1 = 0

ngpu = 1

lambda_gp = 10

d_ratio = 1

img_batch_size = [(4,16),(8,16),(16,16),(32,16),(64,16),(128,16), (256, 14), (512, 6), (1024, 3)]

betas = (0, 0.99)

steps = 800000

small_penalty_e = .001

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# https://github.com/hukkelas/progan-pytorch/blob/master/src/models/custom_layers.py
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        div = torch.square(x)
        div = torch.mean(div, dim = 1, keepdim = True)
        div = div + 10**(-8)
        div = torch.square(div)
        return x/div

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
#         print(std.shape)
        return std
# https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py
class conv2d_e(nn.Module):
    def __init__(self, input_c, output_c, kernel, stride, pad):
        super(conv2d_e, self).__init__()
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(output_c, input_c, kernel, kernel)))
        self.bias = torch.nn.Parameter(torch.FloatTensor(output_c).fill_(0))
        self.stride = stride
        self.pad = pad
        fan_in = (kernel*kernel) * input_c
        self.scale = np.sqrt(2) / np.sqrt(fan_in)
#         print(self.weight.shape)

    def forward(self, x):
        return nn.functional.conv2d(input = x, 
                         weight = self.weight * self.scale, 
                         stride = self.stride, 
                         bias = self.bias, 
                         padding = self.pad)
        

class linear_e(nn.Module):
    def __init__(self, input_c, output_c):
        super(linear_e, self).__init__()
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(output_c, input_c)))
        self.bias = torch.nn.Parameter(torch.FloatTensor(output_c).fill_(0))
        fan_in = input_c
        self.scale = np.sqrt(2) / np.sqrt(fan_in)
#         print(self.weight.shape)

    def forward(self, x):
        return nn.functional.linear(input = x, 
                         weight = self.weight * self.scale, 
                         bias = self.bias)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
#         self.added = nn.ModuleList([])
        self.up_samp = nn.Upsample(scale_factor = 2)
#         4
        self.start = linear_e(512, 8192)
        self.block = nn.Sequential(
            conv2d_e(nz, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end = conv2d_e(512, 3, 1, 1, 0)
        
#         8
        self.block1 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),) 
        self.end1 = conv2d_e(512, 3, 1, 1, 0)
        
#         16
        self.block2 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end2 = conv2d_e(512, 3, 1, 1, 0)
        
#         32
        self.block3 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end3 = conv2d_e(512, 3, 1, 1, 0)
        
#         64
        self.block4 = nn.Sequential(
            self.up_samp,
            conv2d_e(512, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(256, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end4 = conv2d_e(256, 3, 1, 1, 0)
        
#          128
        self.block5 = nn.Sequential(
            self.up_samp,
            conv2d_e(256, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(128, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end5 = conv2d_e(128, 3, 1, 1, 0)
        
#          256
        self.block6 = nn.Sequential(
            self.up_samp,
            conv2d_e(128, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(64, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end6 = conv2d_e(64, 3, 1, 1, 0)

#          512
        self.block7 = nn.Sequential(
            self.up_samp,
            conv2d_e(64, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(32, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end7 = conv2d_e(32, 3, 1, 1, 0)

#          1024
        self.block8 = nn.Sequential(
            self.up_samp,
            conv2d_e(32, 16, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),
            conv2d_e(16, 16, 3, 1, 1),
            nn.LeakyReLU(.2),
            PixelNorm(),)
        self.end8 = conv2d_e(16, 3, 1, 1, 0)

#     Get this down to one if statement logic for fade in
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
            
#         print(output.shape) 
        return output


# Implement batchstdev
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
#         self.added = nn.ModuleList([])
        self.down_samp = nn.AvgPool2d(2)
    
#         4
        self.block = nn.Sequential(
            MiniBatchSTD(),
            conv2d_e(513, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 4, 1, 0),
            nn.LeakyReLU(.2),
            nn.Flatten(),
            linear_e(512, 1))
        self.start = conv2d_e(3, 512, 1, 1, 0)
        
#         8
        self.block1 = nn.Sequential(
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start1 = conv2d_e(3, 512, 1, 1, 0)

#         16
        self.block2 = nn.Sequential(
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start2 = conv2d_e(3, 512, 1, 1, 0)
        
#         32
        self.block3 = nn.Sequential(
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(512, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start3 = conv2d_e(3, 512, 1, 1, 0)
        
#         64
        self.block4 = nn.Sequential(
            conv2d_e(256, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(256, 512, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start4 = conv2d_e(3, 256, 1, 1, 0)
        
#         128
        self.block5= nn.Sequential(
            conv2d_e(128, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(128, 256, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start5 = conv2d_e(3, 128, 1, 1, 0)

#         256
        self.block6= nn.Sequential(
            conv2d_e(64, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(64, 128, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start6 = conv2d_e(3, 64, 1, 1, 0)

#         512
        self.block7= nn.Sequential(
            conv2d_e(32, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            conv2d_e(32, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            self.down_samp,)
        self.start7 = conv2d_e(3, 32, 1, 1, 0)

#         1024
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
            
#         print(output.shape) 
        return output


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
    print("Data Loaded")
    return data_loaders

# https://discuss.pytorch.org/t/copy-weights-only-from-a-networks-parameters/5841
def update_running_avg(original, copy):
    with torch.no_grad(): 
        params1 = original.named_parameters()
        params2 = copy.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_((1 - .999) * dict_params2[name1] + (.999) * param1.data)

def startup(load_train, mixed_precision):
    
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
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
    
    if load_train:
    
        if not mixed_precision:
            pathD = save_directory + "Regular/" + "D/"
            pathG = save_directory + "Regular/" + "G/"
        else:
            pathD = save_directory + "Amp/" + "D/"
            pathG = save_directory + "Amp/" + "G/"
        try:  
            list_of_files = glob.glob(pathD + '*') # * means all if need specific format then *.csv
            latest_file = max(list_of_files)
            checkD = torch.load(latest_file)

            list_of_files = glob.glob(pathG + '*') # * means all if need specific format then *.csv
            latest_file = max(list_of_files)
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
        except:
            pass

    return [netD, netG, netG_copy, optimizerD, optimizerG, scalerD, scalerG, epoch, res, current_data, fade_in, fixed_noise, training]

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

def small_penalty(netD, output_real, mixed_precision = False):
    if mixed_precision:
        pass
    else:
        penalty = torch.square(output_real)
        penalty = penalty * small_penalty_e
    return penalty

def logging(epoch, res, fade_in, count, alpha, loss_D, loss_G, netG_copy, fixed_noise, mixed_precision):
    with open(log_directory + "log.txt","a+") as f:
        print("Res:", res, "Fade_in:", fade_in, "Iter:",count, "alpha:", alpha, file=f)
        print("W with GP:", loss_D.item(),  "Loss G:", loss_G.item(), file=f)
        print(file=f)
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

def save_models(epoch, res, current_data, fade_in, netD, optimizerD, netG, netG_copy, optimizerG, fixed_noise, scalerD, scalerG, training, mixed_precision):
    if not mixed_precision:
        
        pathD = save_directory + "Regular/" + "D/" + "next_res:" + str(res) + "next_fade:" + str(fade_in)
        pathG = save_directory + "Regular/" + "G/" + "next_res:" + str(res) + "next_fade:" + str(fade_in)
        torch.save({
            'training': training,
            'next_epoch': epoch,
            'next_res': res,
            'next_dict': current_data,
            'next_fade': fade_in,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            }, pathD)

        torch.save({
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'fixednoise': fixed_noise,
            'copy_model_state_dict': netG_copy.state_dict(),
            }, pathG)
            
    else:
    
        pathD = save_directory + "Amp/" + "D/" + str(epoch)
        pathG = save_directory + "Amp/" + "G/" + str(epoch)
               
        torch.save({
            'training': training,
            'next_epoch': epoch,
            'next_res': res,
            'next_dict': current_data,
            'next_fade': fade_in,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'scaler_state_dict': scalerD.state_dict(),
#                 'amp': amp.state_dict(),
            }, pathD)

        torch.save({
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'fixednoise': fixed_noise,
            'scaler_state_dict': scalerG.state_dict()

            }, pathG)

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

def training(load_train = False, mixed_precision = False):
    with open(log_directory + "log.txt","a+") as f:
        print("Loaded Models", file=f)
        print(file=f)
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
        data_loaders = load_data()

        print("Starting Training", file=f)
    while(training):

        loader = iter(data_loaders[current_data])
        count = 0
     
        start_time = time.time()
        
        while count < steps:
            try:
                img = loader.next()
            except StopIteration:
                loader = iter(data_loaders[current_data])
                img = loader.next()
            if not fade_in:
                alpha = -1
               
            else:
                alpha = count/steps
                
            mini_batch = len(img[0])
    
            real_imgs = img[0].to(device)
        
            noise = torch.randn(mini_batch, nz, device=device)
        
            netD.zero_grad()
            
#             Discriminator loss on real images
            if mixed_precision:
                pass
            else:
                output_real = netD(real_imgs, alpha=alpha, res=res).squeeze()

            
#             Discriminator loss on fake images
            if mixed_precision:
                pass   
            else:
                fake_imgs = netG(noise, res=res, alpha=alpha)
                output_fake = netD(fake_imgs.detach(), alpha=alpha, res=res).squeeze()
                
                
                
#             Gradient Penalty
            grad_pen = gradient_penalty(netD, netG, mini_batch, real_imgs, fake_imgs, alpha, res, mixed_precision)
            

#             Extra small penalty
            penalty = small_penalty(netD, output_real, mixed_precision)
                

#             Calculating entire loss and taking step
            if mixed_precision:
                pass
            else:
                loss_D = torch.mean(output_fake - output_real + grad_pen + penalty)
                loss_D.backward()
                optimizerD.step()      
                
            netG.zero_grad()
            
#             Generator loss on created batch
            if mixed_precision:
                pass
            else:
                output = netD(fake_imgs, alpha=alpha, res=res).squeeze()
                loss_G = -torch.mean(output)
                loss_G.backward()
                optimizerG.step()
            
            update_running_avg(netG, netG_copy)

#             Training Stats
                
            count += mini_batch
            if count %5000 <= mini_batch:
                logging(epoch, res, fade_in, count, alpha, loss_D, loss_G, netG_copy, fixed_noise, mixed_precision)

        if fade_in == False:
            fade_in = True
            current_data += 1
            if current_data == len(data_loaders):
                training= False
            res = res * 2
        else:
            fade_in = False
        epoch += 1 
       
    
        save_models(epoch, res, current_data, fade_in, netD, optimizerD, netG, netG_copy, optimizerG, fixed_noise, scalerD, scalerG, training, mixed_precision)

    
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch time: ", epoch_time)

def main(argv):
    arg = int(argv[0])
    if arg > 0:
        size = 16
        if argv == 256:
            size = 14
        elif argv == 512:
            size = 6
        elif argv == 1024:
            size = 3
        print(argv[0])
        fixed_noise = torch.randn(16, nz, device=device)
        noise = torch.randn(size, nz, device=device)
        netG = Generator(ngpu).to(device)
        netD = Discriminator(ngpu).to(device)
        netG_copy = deepcopy(netG)
        test  = netG(noise, alpha = .5, res = arg)
        test1 = netD(test, alpha = .5, res = arg)
        update_running_avg(netG, netG_copy)
        with torch.no_grad():
            test2 = netG_copy(fixed_noise, alpha = .5, res = arg)
        print("success")
        return 0
    
    else:
        check_directories()
        training(load_train = True, mixed_precision=False)



if __name__ == "__main__":
   main(sys.argv[1:])
