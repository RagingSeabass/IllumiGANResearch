
from models.data_loader import LearningToSeeInTheDarkDataset
from models.net import UNet
from models.net_test import LSID
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os, time, sys
from utils import AverageMeter

batch_size = 1
epochs = 1000
learning_rate = 1e-4
save_freq = 250

checkpoint_dir = './checkpoint/sony/'

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

dataset = LearningToSeeInTheDarkDataset("/work3/s164440/shared/Learning-to-See-in-the-Dark/dataset/sony/", debug=True)
#dataset = LearningToSeeInTheDarkDataset("/Users/groenbech/Desktop/Software/Python/LearnToSeeInTheDark/dataset/sony/")
data_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Check if gpu support is available
cuda_avail = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_avail else 'cpu')

#Create model, optimizer and loss function
#model = UNet()
#model._initialize_weights()

model = LSID()
model.train()

criterion = nn.L1Loss()

with open('log_2.txt', 'a') as f:
    f.write("New training \n")

if cuda_avail:
    print("Cuda available", file=sys.stderr, flush=True)
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):

    if epoch > 500:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    
    running_loss = AverageMeter()
    
    st = time.time()
    for short_patch, long_patch in data_generator:
        
        # Transfer to GPU
        if cuda_avail:
            short_patch = short_patch.permute(0,3,1,2).cuda(device=device)
            long_patch = long_patch.permute(0,3,1,2).cuda(device=device)
        else:
            short_patch = short_patch.permute(0,3,1,2).to(device)
            long_patch = long_patch.permute(0,3,1,2).to(device)
        
        model.zero_grad()
        out = model(short_patch)

        loss = criterion(out, long_patch) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item())
        
    if epoch % save_freq == 0:
        torch.save({
            'model_state': model.state_dict(),
            'epoch' : epoch,
            'optimizer_state': optimizer.state_dict()
            }, checkpoint_dir + 'sony_epoch_%04d.pth' % epoch)

    with open('log_2.txt', 'a') as f:
        f.write("%d Loss=%.3f Time=%.3f \n" % (epoch, running_loss.average(), time.time() - st))
    

