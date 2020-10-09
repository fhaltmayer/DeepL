import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Networks import Generator, Discriminator
# Set random seed for reproducibility
# manualSeed = 999
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

dataroot = "/media/fico/Data/Celeba/CelebAMask-HQ"
checkpoint_dir = "./Checkpoint/"
check_name = "celeb_64"

num_workers = 4
batch_size = 128
image_size = 64
nz = 100
nf = 64
nc = 3
num_epochs = 50
lr = 0.0001
betas = (0, .9)
lambda_gp = 10
d_ratio = 5
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class WGan():
    def __init__(self):
        self.netG = Generator().to(device)
        self.netD = Discriminator().to(device)
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        self.fixed_noise = torch.randn(16, nz, 1, 1, device=device)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=betas)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=betas)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def checkpoint(self, epoch):
        path = checkpoint_dir + check_name
        torch.save({
            'netG': self.netG.state_dict(),
            'netD': self.netD.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'fixed_noise': self.fixed_noise,
            # 'scaler_gen': self.scaler_gen.state_dict(),
            # 'scaler_dis': self.scaler_dis.state_dict(),
            'epoch': epoch,
            }, path)

    # Load latest checkpoint
    def loadcheck(self):
        path = checkpoint_dir + check_name
        check = torch.load(path)
        self.netG.load_state_dict(check['netG'])
        self.netD.load_state_dict(check['netD'])
        self.optimizerD.load_state_dict(check['optimizerD'])     
        self.optimizerG.load_state_dict(check['optimizerG'])
        self.fixed_noise = check['fixed_noise']
        # self.scaler_gen.load_state_dict(check['scaler_gen'])          
        # self.scaler_dis.load_state_dict(check['scaler_dis'])
        return check['epoch']    
    
    # Calculate gradient penalty described in the paper
    def gradient_penalty(self):
        alpha = torch.randn(batch_size, 1, 1, 1, device = device)
        interp = alpha * self.real_batch + ((1-alpha) * self.fake_batch.detach())
        interp.requires_grad_()

        model_interp = self.netD(interp)
        grads = torch.autograd.grad(outputs=model_interp, inputs=interp,
                                  grad_outputs=torch.ones(model_interp.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads = torch.square(grads)
        grads = torch.sum(grads, dim = [1,2,3])
        grads = torch.sqrt(grads)
        grads = grads - 1
        grads = torch.square(grads)
        grad_pen = torch.mean(grads * lambda_gp)
        return grad_pen

    # Calculating the discriminator loss
    def dis_loss(self):
        loss_real = self.netD(self.real_batch)
        loss_real = -torch.mean(loss_real)

        loss_fake = self.netD(self.fake_batch.detach())
        loss_fake = torch.mean(loss_fake)
        
        grad_pen = self.gradient_penalty()
        
        d_loss = loss_fake + loss_real + grad_pen
        return d_loss

    def gen_loss(self):
            g_loss = self.netD(self.fake_batch)
            g_loss = -torch.mean(g_loss)
            return g_loss

    # One training step where the discriminator is trained everytime while the generator is trained every x batches
    def step(self, real_batch, epoch, i):
        self.real_batch = real_batch
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        self.fake_batch = self.netG(noise)

        self.optimizerD.zero_grad()
        d_loss = self.dis_loss()
        d_loss.backward()
        self.optimizerD.step()

        if i % d_ratio == 0:
            self.optimizerG.zero_grad()
            g_loss = self.gen_loss()
            g_loss.backward()
            self.optimizerG.step()
            if i % 20 == 0:
                self.log(g_loss.item(), d_loss.item(), epoch, i)

    # Simple logging to stdout and a plt figure
    def log(self, g_loss, d_loss, epoch, i):
        plt.close('all')
        print("\nEpoch:", epoch, "Iteration:", i * batch_size)
        print("Discriminator Loss:", d_loss, "G Loss:", g_loss)
        with torch.no_grad():
            guess = self.netG(self.fixed_noise)
            guess = guess.cpu()
            old_min = -1
            old_max = 1
            old_range = old_max - old_min
            new_range = 1 - 0
            guess = (((guess - old_min)*new_range)/ old_range) + 0
            guess = guess.permute(0,2,3,1)
            fig = plt.figure(figsize=(4,4))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(guess[i, :, :])
                plt.axis('off')
            
            plt.show(block=False)
            plt.pause(0.001)

def get_data():
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    return dataloader

def train():
    print("Train Start")
    wgan = WGan()
    epo = 0
    dataloader = get_data()
    try:
        epo = wgan.loadcheck()
        print("Loaded")
        epo += 1
    except:
        print("Failed to load")

    for epoch in range(epo, num_epochs):
        for i, data in enumerate(dataloader, 0):
            wgan.step(data[0].to(device), epoch, i)
        wgan.checkpoint(epoch)
    print("Done Training")


def check_directories():
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

def main():
    check_directories()
    train()

if __name__ == "__main__":      
    main()
