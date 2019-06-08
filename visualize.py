
# Load visualization into csv
import re
import matplotlib.pyplot as plt

files = ['../../../expmobile/train1.log', '../../../expmobile/train2.log', '../../../expmobile/train3.log']
#lines = [line.rstrip('\n') for line in open(file)]

epochs = []
lossGs = []
lossDs = []
averageTime = 0
totalIteartions = 0
startTime = ""
times = []
first = True
for file in files:
    iterations = 0
    lines = [line.rstrip('\n') for line in open(file)]
    for line in lines:
         
        if first:
            # Get time and date of entry
            startTime = re.search(r'\d{4}-\d{2}-\d{2}', line).group()
            # time = re.search(r'\d{2}[":"]\d{2}[":"]\d{2}', line).group()
            first = False

        epoch = re.search(r'Epoch [0-9]*', line)
        if (epoch):
            epoch = epoch.group()[6:] # Remove text
            epochs.append(float(epoch))
            lossG = re.search(r'Loss G: [0-9]*["."][0-9]*[e]*-*[0-9]*', line).group()[7:]
            lossGs.append(float(lossG))
            lossD = re.search(r'D: [0-9]*["."][0-9]*[e]*-*[0-9]*', line).group()[3:]
            lossDs.append(float(lossD))
            epochTime =  re.search(r'Time [0-9]*["."][0-9]{3}', line).group()[5:]
            times.append(float(epochTime))
            iterations = re.search(r'Iteration [0-9]*', line).group()[10:]

    totalIteartions += int(iterations)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
ax1 = plt.subplot(121)
plt.subplots_adjust(0.1,0.1,0.9,0.9,0.3,0.2)
p1 = ax1.plot(epochs, lossGs, 'royalblue', label='train loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Gen loss (train)')

ax2 = ax1.twinx()
p2 = ax2.plot([0, 1000, 2000, 3000, 4000], [0.5, 0.090, 0.101, 0.100, 0.097], 'firebrick', label='test loss')
ax2.set_ylabel('Gen L1 loss (test)')

# added these three lines
lns = p1+p2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0)

plt.title('Generator loss')


plt.subplot(122)
plt.plot(epochs, lossDs, 'royalblue')

plt.xlabel('Epochs')
plt.ylabel('GANloss')
plt.title('Discriminator loss')

plt.show()
fig.savefig('lossmobile.png')


