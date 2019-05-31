
# Load visualization into csv
import re
import matplotlib.pyplot as plt

files = ['../../../exp2/train.log']
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


plt.figure(1)

ax1 = plt.subplot(121)
ax1.plot(epochs, lossGs, 'b', label='train loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('L1 loss (train)')

ax2 = ax1.twinx()
ax2.plot([0, 1000, 1500, 2000, 2500, 3000, 3500, 4000], [1, 0.056, 0.046, 0.05, 0.05, 0.05, 0.05, 0.05], 'r', label='test loss')
ax2.set_ylabel('L1 loss (test)')

plt.legend()
plt.title('Generator loss')


plt.subplot(122)
plt.plot(epochs, lossDs)

plt.xlabel('Epochs')
plt.ylabel('GANloss')
plt.title('Discriminator loss')

plt.show()



