
from models.data_loader import LearningToSeeInTheDarkDataset
from models.net import UNet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os, time

batch_size = 1
epochs = 1000
learning_rate = 1e-4
save_freq = 250

checkpoint_dir = './checkpoint/sony/'

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

dataset = LearningToSeeInTheDarkDataset("/work3/s164440/shared/Learning-to-See-in-the-Dark/dataset/sony/")
#dataset = LearningToSeeInTheDarkDataset("/Users/groenbech/Desktop/Software/Python/LearnToSeeInTheDark/dataset/sony/")
data_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Check if gpu support is available
cuda_avail = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_avail else 'cpu')

#Create model, optimizer and loss function
model = UNet()


with open('log.txt', 'a') as f:
    f.write("New training \n")

# Move model before constructing optimizer 
if torch.cuda.device_count() > 1:
    print("Running in parrallel: " + str(torch.cuda.device_count()) + " GPU's",file=sys.stderr, flush=True )
    model = nn.DataParallel(model,  device_ids=[0, 1]).cuda()
else if cuda_avail:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):

    if epoch > 500:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    
    running_loss = []

    st = time.time()
    for short_patch, long_patch in data_generator:
        
        # Transfer to GPU
        short_patch = short_patch.permute(0,3,1,2).to(device)
        long_patch = long_patch.permute(0,3,1,2).to(device)
        
        model.zero_grad()
        out = model(short_patch)

        loss = torch.abs(out - long_patch).mean()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    if epoch % save_freq == 0:
        torch.save({
            'model_state': model.state_dict(),
            'epoch' : epoch,
            'optimizer_state': optimizer.state_dict()
            }, checkpoint_dir + 'sony_epoch_%04d.pth' % epoch)

    
    with open('log.txt', 'a') as f:
        f.write("%d Loss=%.3f Time=%.3f \n" % (epoch, sum(running_loss)/len(running_loss), time.time() - st))
    

