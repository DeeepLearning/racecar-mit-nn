"""
Script for training network with alexnet architecture.
25.01.18 - Tonio
"""
# ''' -- Python 3 nifty things '''
from __future__ import division
from __future__ import print_function
# from builtins import *

# ''' -- imports: '''
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim                   # For optimizing
import torch.nn as nn
import numpy as np
import os # to grab the name of the classes
from os.path import normpath, basename #to extract the name of the images
from torch.autograd import Variable
import sys # to modify path and include custom modules.
import datetime #for output file name


# ''' -- custom imports: '''
from networks import incrementalAlexNet


#### Script Parameters ---------

# --------------------------------
#         Choosing Datasets
# --------------------------------


##################################
# --------- Loader params: -------
imageSize = 128 # Full size is 128
batchSize = 300 # 
shuffle = True  #
numWorkers = 2  #
epochNum = 20    # number of epochs
#### -----------------------------




# --------------------------------
#           Data Loader
# --------------------------------
# create transformations:
transformsArray = createDataAugmentationTransforms(imageSize)
# create DataLoader (using just the original dataset, no data augmentation)
dataAugTrainLoader = createDataAugmentedDataset(trainSetDir, transformsArray, batchSize, shuffle, numWorkers)
# obtain the class names from the directories:
classNames = os.listdir(trainSetDir)
if classNames[0] == '.DS_Store':
	classNames.pop(0)




# --------------------------------
#        Create the Network
# --------------------------------
# Create a network:
net = incrementalAlexNet(batch_size=batchSize)

# --------------------------------
#    Choose your loss function
# --------------------------------
# Define a Loss function
criterion = nn.CrossEntropyLoss()
learningRate = 0.07

# --------------------------------
#            Training
# --------------------------------
# Train the network
for epoch in range(epochNum): # need to loop ever the dataset multiple times
	print("Entering epoch: %d" % (epoch))
	# ---------------------------
	#   Learning Rate Scheduling
	# ---------------------------	
	if epoch == 1:
		learningRate = 0.1
	elif epoch == 2:
		learningRate = 0.1
	else:
		learningRate = 0.1
	# ---------------------------
	#   Create the Optimizer
	# ---------------------------
	optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum = 0.9)
	running_loss = 0.0
	# ---------------------------
	#   Loooooooooooooooooop!
	# ---------------------------
	for i, data in enumerate(dataAugTrainLoader, 0):
		# get the inputs
		inputs, labels = data
		# wrap them in Variable
		inputs, labels = Variable(inputs), Variable(labels)
		# zero the parameter gradients
		optimizer.zero_grad()
		# adjust for GPU use
		if torch.cuda.is_available():
			inputs, labels = inputs.cuda(), labels.cuda()
			net.cuda()
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.data[0]
		if i%100 == 99: # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
			running_loss = 0.0

print('Finished Training')

# --------------------------------
#            Validation
# --------------------------------
# Set the model in evaluation mode
net.eval()
# normalization transformation
transform1 = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#Load the validation set
valset = myImageFolder(root = valSetDir , transform = transform1)
valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle = False, num_workers=2)
#Map the validation set names to categories
Map = {}
with open( rootDir + 'val.txt') as text_file:
	content = text_file.readlines()
content = [x.rstrip('\n') for x in content]
for row in content:
	Map[row[4:16]] = int(row[17:19])
#Get stats on how well the network performs
correct_1 = 0
correct_5 = 0
total = 0
for data in valloader:
	valimages, vallabels, valImgNames = data
	if torch.cuda.is_available():
		valimages, vallabels = valimages.cuda(), vallabels.cuda()
	valoutput = net(Variable(valimages))
	scores,predicted = torch.topk(valoutput.data, 5, 1)
	if torch.cuda.is_available():
		p = predicted.cpu().numpy().tolist()
	else:
		p = predicted.numpy().tolist()

	#Assess top 1 performance
	total += vallabels.size(0)
	for i in range(len(p)):
		correct_1 += (p[i][0] == Map[valImgNames[i]])
		if [Map[valImgNames[x]] for x in range(10)][i] in p[i]:
			correct_5 +=1

print('Top 1 Accuracy of the network on the 10000 test images: %f %%' % (
	100 * correct_1 / total))

print('Top 5 Accuracy of the network on the 10000 test images: %f %%' % (
	100 * correct_5 / total))










# Load the test set:
transform1 = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = myImageFolder(root=testSetDir, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle = False, num_workers=2)

predictions = []
imageFilenames = []
for testdata in testloader:
	testimages, testlabels, testImgNames = testdata
	if torch.cuda.is_available():
		testimages, testlabels = testimages.cuda(), testlabels.cuda()
	testoutput = net(Variable(testimages))
	testscores, testpredicted = torch.topk(testoutput.data, 5, 1)

	if torch.cuda.is_available():
		testp = testpredicted.cpu().numpy().tolist()
	else:
		testp = testpredicted.numpy().tolist()

	for i in range(len(testp)):
		predictions.append(testp[i])
		imageFilenames.append(testImgNames[i])

# create .txt
outputFile = open("Output-currentAlexFull_" + datetime.datetime.now().strftime("%I:%M-%B-%d-%Y") + ".txt", "w")
for i in range(len(predictions)):
	outputFile.write("test/" + imageFilenames[i] + " %d %d %d %d %d\n" %(predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3], predictions[i][4]))
outputFile.close()



print("*****************\n****   DONE!!!! \n*********************\n\n")


