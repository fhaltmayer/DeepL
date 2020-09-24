import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
# plt.ion()
dataroot = "/media/fico/Data/Celeba/CelebAMask-HQ"
save_path = "./Pytorch_checkpoint/"
img_size = 64
batch_size = 256
num_workers = 4
num_epochs = 50
feat = 32
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("Device", device)


# Simple data loader that crops each image to the right size and sets range to [0,1]
def load_data():
    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                                transform=transforms.Compose([
                                                           transforms.Resize(img_size),
                                                           transforms.CenterCrop(img_size),
                                                           transforms.ToTensor(),
                                                           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))

    dataload = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, drop_last=True)
    return dataload

# Initializing weights, currently set to he's equivalent but unused due getting stuck training.
def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

# Just a basic autoencoder, commented out code is different style of blocks that can be used, although this had
# the best results so far
class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            
            torch.nn.Conv2d(3, feat, 3, 2, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.Tanh(),
            # torch.nn.MaxPool2d(2),
            # torch.nn.Conv2d(feat, feat, 4, 2, 1),
            # torch.nn.ReLU(),
            
            torch.nn.Conv2d(feat, feat*2, 3, 2, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.Tanh(),
            # torch.nn.MaxPool2d(2),
            # torch.nn.Conv2d(feat*2, feat*2, 4, 2, 1),
            # torch.nn.ReLU(),

            
            torch.nn.Conv2d(feat*2, feat*4, 3, 2, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.Tanh(),
            # torch.nn.MaxPool2d(2), 
            # torch.nn.Conv2d(feat*4, feat*4, 4, 2, 1),
            # torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8192,3072),
      
            )

        self.decoder_lin = torch.nn.Linear(3072,8192)

        self.decoder = torch.nn.Sequential(
            # torch.nn.Conv2d(feat*8, feat*4, 3, 1, 1),
            # torch.nn.ReLU(),
            # torch.nn.UpsamplingBilinear2d(scale_factor = 2),
            torch.nn.UpsamplingBilinear2d(scale_factor = 2),
            torch.nn.Conv2d(feat*4, feat*2, 3, 1, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.Tanh(),
            # torch.nn.ConvTranspose2d(feat*2, feat*2, 4, 2, 1),
            # torch.nn.ReLU(),
            torch.nn.UpsamplingBilinear2d(scale_factor = 2),
            torch.nn.Conv2d(feat*2, feat, 3, 1, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.Tanh(),
            # torch.nn.ConvTranspose2d(feat, feat, 4, 2, 1),
            # torch.nn.ReLU(),
            torch.nn.UpsamplingBilinear2d(scale_factor = 2),
            torch.nn.Conv2d(feat, 3, 3, 1, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.Tanh(),
            torch.nn.Conv2d(3, 3, 1, 1, 0),
            torch.nn.LeakyReLU(),

            # torch.nn.Tanh()
            torch.nn.Sigmoid()
            )

    # Different arguments for only enocder and only decoder usage
    def forward(self, input, stage = -1):
        if stage == -1:
            output = self.encoder(input)
            output = self.decoder_lin(output)
            output = output.view(-1,128,8,8)
            output = self.decoder(output)
        
        elif stage == 1:
            output = self.encoder(input)

        else:
            output = self.decoder_lin(input)
            output = output.view(-1,128,8,8)
            output = self.decoder(output)
        
        return output

# Simple checkpointing of the model, overwrites at each checkpoint call
def checkpoint(a, e, o):
    path = save_path + "check"
    torch.save({
    'ae_state_dict': a.state_dict(),
    'optimizer_state_dict': o.state_dict(),
    'epoch': e
    }, path)

# Check if the directories being used exist 
def check_directories():
    if not os.path.exists(save_path):
            os.makedirs(save_path)

# Printing out 10 images before and after encoding/decoding
def image_log(real, output):
    # old_min = -1
    # old_max = 1
    # old_range = old_max - old_min
    # new_range = 1 - 0
    # real = (((real - old_min)*new_range)/ old_range) + 0
    # output = (((output - old_min)*new_range)/ old_range) + 0
    plt.close("all")
    real = real.permute(0,2,3,1)
    output = output.permute(0,2,3,1)
    fig= plt.figure(figsize=(2,10))
    col = 2
    row = 10
    count = 0
    for i in range(1, col * row + 1, 2):
        if 1%2 ==1:
            img = real[count, :, :]
            fig.add_subplot(row, col, i)
            plt.axis('off')
            plt.imshow(img)
            img = output[count, :, :]
            fig.add_subplot(row, col, i + 1)
            plt.axis('off')
            plt.imshow(img) 
            count += 1
    plt.show(block=False)
    plt.savefig("./test.png")
    plt.pause(0.001)

# Training loop
def train():
    ae = AE().to(device)
    # ae.apply(init_weights)
    optimizer = torch.optim.Adam(ae.parameters())
    crit = torch.nn.BCELoss()
    epoch = 0
    
    try:
        check = torch.load(save_path + "check")
        optimizer.load_state_dict(check['optimizer_state_dict'])
        ae.load_state_dict(check['ae_state_dict'])
        epoch = check['epoch']
    
    except:
        print("Starting new training")
        
    data = load_data()
    steps = len(data)    
    while epoch < num_epochs:
        for ite, (inputs, _) in enumerate(data):
            inputs = inputs.to(device)
            ae.zero_grad()
            output = ae(inputs)
            loss = crit(output, inputs)
            loss.backward()
            optimizer.step()
            print("Epoch:", epoch, "Step:", str(ite) + "/" + str(steps), "Loss:", loss.item())
            if ite%20 == 0:
                image_log(inputs.cpu(), output.detach().cpu())
        
        epoch += 1
        checkpoint(ae, epoch, optimizer)
        
def main():
    check_directories()      
    train()


if __name__ == "__main__":
    main()
        
        



