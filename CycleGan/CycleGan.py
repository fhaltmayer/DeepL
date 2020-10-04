import torch
from Networks import Generator, Discriminator, ImageBuffer
from Data import DataLoad
import torch.nn as nn
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
lamb = 10
lr = .0002
ident_weight = .5
batch_size = 4
img_size = 283
crop_size = 256
num_workers = 4
epochs = 200
scale_epoch = 100
log_percent = .05
betas = (.5, .999)
checkpoint_dir = "./Checkpoint/"
check_name = "ukiyoe_check_283x256"

x_root = "/media/fico/Data/Kaggle/ukiyoe2photo/trainB/"
y_root = "/media/fico/Data/Kaggle/ukiyoe2photo/trainA/"



# x: input
# y: target
# gen_y: x -> y
# gen_x: y -> x
# dis_x: x vs gen_x(y)
# dis_y: y vs gen_y(x)
class CycleGan:
    def __init__(self, rb=9):
        self.gen_x = Generator(rb).to(device)
        self.gen_y = Generator(rb).to(device)
        self.dis_x = Discriminator().to(device)
        self.dis_y = Discriminator().to(device)
        self.fake_x_buffer = ImageBuffer()
        self.fake_y_buffer = ImageBuffer()
        self.crit = nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.optimizer_gen = torch.optim.Adam(list(self.gen_x.parameters()) + list(self.gen_y.parameters()), lr = lr, betas = betas)
        self.optimizer_dis = torch.optim.Adam(list(self.dis_x.parameters()) + list(self.dis_y.parameters()), lr = lr, betas = betas)
        self.scaler_gen = torch.cuda.amp.GradScaler()
        self.scaler_dis = torch.cuda.amp.GradScaler()
        self.lr_dis = None
        self.lr_gen = None
        
        self.gen_y.apply(self.init_weights)
        self.gen_x.apply(self.init_weights)
        self.dis_x.apply(self.init_weights)
        self.dis_y.apply(self.init_weights)
    
    # Initiate weights with normal distribution
    def init_weights(self, m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.normal_(m.weight, std=0.02, mean = 0.0)
            
    # Toggle gradient tracking on discriminators
    def grad_toggle(self, grad):
        for param in self.dis_x.parameters():
            param.requires_grad = grad
        for param in self.dis_y.parameters():
            param.requires_grad = grad

    # Generator loss is gen_a vs 1's
    def loss_gen(self, result):
        return self.crit(result, torch.ones_like(result).to(device))
    
    # Dicriminator loss is gen_a vs a
    def loss_dis(self, real, fake):
        loss1 = self.crit(real, torch.ones_like(real).to(device))
        loss2 = self.crit(fake, torch.zeros_like(fake).to(device))
        return (loss1 + loss2) * .5

    # Cyclic loss is a vs gen_a(gen_b(a))
    def loss_cyclic(self, cycled, real):
        loss = self.l1(cycled, real) * lamb
        return loss

    # Identity loss a vs gen_a(a)
    def loss_identity(self, ident, real):
        loss = self.l1(ident, real) * lamb * ident_weight
        return loss
    
    # Return a the generated and cycled image
    def test_photo(self, image, y_in = False):
        if y_in == False:
            with torch.no_grad():
                fake_y = self.gen_y(image)
                cycled = self.gen_x(fake_y)
                return (fake_y, cycled)
        else:
            with torch.no_grad():
                fake_x = self.gen_x(image)
                cycled = self.gen_y(fake_x)
                return (fake_x, cycled)

    def step(self, x, y, step, total_step, log = False): 
        self.optimizer_gen.zero_grad()
        self.grad_toggle(False)

        # Finding loss of the generators
        with torch.cuda.amp.autocast():
            # input -> target
            fake_y = self.gen_y(x)
            output_fake_y = self.dis_y(fake_y)
            loss_fake_y = self.loss_gen(output_fake_y)
    
            # target -> input
            fake_x = self.gen_x(y)
            output_fake_x = self.dis_x(fake_x)
            loss_fake_x = self.loss_gen(output_fake_x)

            # cycled
            cycled_y = self.gen_y(fake_x)
            loss_cycled_y = self.loss_cyclic(cycled_y, y)

            cycled_x = self.gen_x(fake_y)
            loss_cycled_x = self.loss_cyclic(cycled_x, x)
            
            # identities
            ident_x = self.gen_x(x)
            ident_y = self.gen_y(y)

            loss_ident_y = self.loss_identity(ident_y, y)
            loss_ident_x = self.loss_identity(ident_x, x)

            loss_g = loss_fake_y + loss_cycled_x + loss_fake_x + loss_cycled_y + loss_ident_y + loss_ident_x
        
        self.scaler_gen.scale(loss_g).backward()
        self.scaler_gen.step(self.optimizer_gen)
        self.scaler_gen.update()
        self.grad_toggle(True)
        self.optimizer_dis.zero_grad()

        # Finding loss of the discriminator
        with torch.cuda.amp.autocast():
            temp = self.fake_y_buffer.grab(fake_y.detach())
            dis_fake_y = self.dis_y(temp.detach())
            dis_y = self.dis_y(y)
            loss_Dy = self.loss_dis(dis_y, dis_fake_y)
            
            temp = self.fake_x_buffer.grab(fake_x.detach())
            dis_fake_x= self.dis_x(temp.detach())
            dis_x = self.dis_x(x)
            loss_Dx = self.loss_dis(dis_x, dis_fake_x)
            loss = loss_Dx + loss_Dy
            
        self.scaler_dis.scale(loss).backward()
        self.scaler_dis.step(self.optimizer_dis)
        self.scaler_dis.update()
        
        if log:
            print("Step:", str(step) + "/" + str(total_step),)
            print("loss fake_y: ", loss_fake_y.item(), "loss cycled_x:", loss_cycled_x.item(), "loss ident_x:", loss_ident_x.item())
            print("loss fake_x: ", loss_fake_x.item(), "loss cycled_y:", loss_cycled_y.item(), "loss ident_y:", loss_ident_y.item())
            print("loss Dx:",  loss_Dx.item(), "loss Dy:",  loss_Dy.item())
        
    # Checkpoint the training
    def checkpoint(self, epoch):
        path = checkpoint_dir + check_name
        torch.save({
            'gen_x': self.gen_x.state_dict(),
            'gen_y': self.gen_y.state_dict(),
            'dis_x': self.dis_x.state_dict(),
            'dis_y': self.dis_y.state_dict(),
            'optimizer_gen': self.optimizer_gen.state_dict(),
            'optimizer_dis': self.optimizer_dis.state_dict(),
            'scaler_gen': self.scaler_gen.state_dict(),
            'scaler_dis': self.scaler_dis.state_dict(),
            'lr_gen': self.lr_gen,
            'lr_dis': self.lr_dis,
            'epoch': epoch,
            }, path)

    # Load latest checkpoint
    def loadcheck(self):
        path = checkpoint_dir + check_name
        check = torch.load(path)
        self.gen_x.load_state_dict(check['gen_x'])
        self.gen_y.load_state_dict(check['gen_y'])
        self.dis_x.load_state_dict(check['dis_x'])
        self.dis_y.load_state_dict(check['dis_y'])
        self.optimizer_gen.load_state_dict(check['optimizer_gen'])     
        self.optimizer_dis.load_state_dict(check['optimizer_dis'])
        self.scaler_gen.load_state_dict(check['scaler_gen'])          
        self.scaler_dis.load_state_dict(check['scaler_dis'])
        try:
            self.self.lr_gen = check['lr_gen']
            self.self.lr_dis = check['lr_dis']
        except:
            pass
        return check['epoch']    

    # Start linearly scaling optimizer at scale_epoch
    def scale_optimizers(self, current_epoch, total_epochs, scale_epoch):
        if current_epoch< scale_epoch:
            pass
        else:
            if self.lr_dis == None:
                self.lr_dis = self.optimizer_dis.param_groups[0]["lr"]
                self.lr_gen = self.optimizer_gen.param_groups[0]["lr"]
            scale = 1 - (current_epoch - scale_epoch)/(total_epochs-scale_epoch)
            self.optimizer_dis.param_groups[0]["lr"] = self.lr_dis * scale
            self.optimizer_gen.param_groups[0]["lr"] = self.lr_gen * scale

# Log training image progress
def image_log(x, y, fake_x, fake_y, cyc_x, cyc_y):
    plt.close("all")
    old_min = -1
    old_max = 1
    old_range = old_max - old_min
    new_range = 1 - 0
    x = (((x - old_min)*new_range)/ old_range) + 0
    y = (((y - old_min)*new_range)/ old_range) + 0
    fake_x = (((fake_x - old_min)*new_range)/ old_range) + 0
    fake_y = (((fake_y - old_min)*new_range)/ old_range) + 0
    cyc_x = (((cyc_x - old_min)*new_range)/ old_range) + 0
    cyc_y = (((cyc_y - old_min)*new_range)/ old_range) + 0
    x = x.permute(0,2,3,1)
    y = y.permute(0,2,3,1)
    fake_x = fake_x.permute(0,2,3,1)
    fake_y = fake_y.permute(0,2,3,1)
    cyc_x = cyc_x.permute(0,2,3,1)
    cyc_y = cyc_y.permute(0,2,3,1)
    fig= plt.figure(figsize=(10,50))
    col = 3
    row = 2
    count = 0
    
    img = x[count, :, :]
    fig.add_subplot(2, 3, 1).set_title('input')
    plt.imshow(img)
    plt.axis('off')
            
    img = y[count, :, :]
    fig.add_subplot(2, 3, 4)
    plt.imshow(img) 
    plt.axis('off')

    img = fake_x[count, :, :]
    fig.add_subplot(2, 3, 2).set_title('target')
    plt.imshow(img)
    plt.axis('off')
            
    img = fake_y[count, :, :]
    fig.add_subplot(2, 3, 5)
    plt.imshow(img) 
    plt.axis('off')

    img = cyc_x[count, :, :]
    fig.add_subplot(2, 3, 3).set_title('cycled')
    plt.imshow(img)
    plt.axis('off')
            
    img = cyc_y[count, :, :]
    fig.add_subplot(2, 3, 6)
    plt.imshow(img) 
    plt.axis('off')
            

    plt.show(block=False)
    plt.pause(0.001)

# Check if directories exist, if not create them
def check_directories():
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)




def main(arg = 0):
    epo = 0
    check_directories()
    images = DataLoad(x_root, 
                  y_root, 
                  batch_size=batch_size, 
                  num_workers=num_workers,
                  img_size=img_size, 
                  crop_size=crop_size, 
                  zero_pad = 6,
                  fast_load = False)
    gan = CycleGan()
    
    try:
        epo = int(gan.loadcheck())
        epo = epo+1
    except:
        pass
    # Training loop
    if arg == 0:
        print("Starting Training: " + str(images.size_x))
        count = max(images.size_x, images.size_y)
        log_count = int(count * log_percent)
        for epoch in range(epo, epochs):
            print("Starting Epoch: "  + str(epoch))
            gan.scale_optimizers(epoch, epochs, scale_epoch)
            for t in range(count):
                x, y = images.next_pair(device=device)
                if t %log_count == 0:
                    gan.step(x, y, t, images.size_x, log = True)
                    temp_x, x_cyc = gan.test_photo(x)
                    temp_y, y_cyc = gan.test_photo(y, y_in = True)
                    image_log(x.cpu(), y.cpu(), temp_x.cpu(), temp_y.cpu(), x_cyc.cpu(), y_cyc.cpu())
                    plt.close('all')
                else:
                    gan.step(x, y,  t, images.size_x)
            gan.checkpoint(epoch)
    
    # Test input
    elif arg ==1:
        im = Image.open("/media/fico/Data/Kaggle/summer2winter_yosemite/trainB/trainB/2008-05-17 11:11:13.jpg")
        print(im.size)
        im = transforms.Resize(256)(im)
        
        # im = transforms.RandomCrop(256)(im)
        im.show()
        im = np.array(im, dtype = np.float32)
        im = im.transpose(2,0,1)
        im = (im  - 127.5) / 127.5  
        print(np.min(im))
        im = torch.from_numpy(im)
        im = im.unsqueeze(0).to(device)
        print(im.shape)
        x, x_cyc = gan.test_photo(im)
        x = x.cpu()
        old_min = -1
        old_max = 1
        old_range = old_max - old_min
        new_range = 1 - 0
        x = (((x - old_min)*new_range)/ old_range) + 0
        # x = x.permute(0,2,3,1)
        x = x.squeeze()
        im = transforms.ToPILImage()(x)
        print(im.size)
        im.show()
        im.save("./desk_test.jpg")
        
if __name__ == "__main__":      
    main(arg = 0)
