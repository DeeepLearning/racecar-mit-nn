"""
File for playing around with
distinct network architectures.

Number of classes for us is: 11
(see README for meaning of the classes)

25.01.18 - tonio
"""

# --- Imports:
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


''' -------------------------------------------------------- '''
'''    Sample network from PyTorch Tutorials (60minBlitz)    '''
''' -------------------------------------------------------- '''

class Net(nn.Module):
	"""
	Sample network from PyTorch Tutorials (60minBlitz)
	"""
	def __init__(self):
		super(Net, self).__init__()
		'''
		- 1 input image (32x32?)
		- 10 output channels
			2 convolutional layers, followed by 3 fully
			connected linear layers.
		'''
		self.conv1 = nn.Conv2d(1, 6, 5)  # 1in,  6out, 5x5 square conv kernel
		self.conv2 = nn.Conv2d(6, 16, 5) # 6in, 16out
		# affine operations: y = Wx + b
		self.fc1 = nn.Linear(16*5*5, 120) # 150in features, 120 out features
		self.fc2 = nn.Linear(120,84)     # 120in features,  84 out features
		self.fc3 = nn.Linear(84,10)      #  84in features,  10 out features

	def forward(self, x):
		'''
		- x is what? I think the input to the forward pass of the network, e.g., an image
		*** x is a tensor! obviously! duh! haha.
		    an autograd.Variable, so that the backward pass can be automatically calculated.
		'''
		# Perform ReLU activation operation after convolution before max pooling
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # max pool 2x2 region after 1st convLayer
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)     # max pool again after 2nd convLayer
		# the view operation returns a new tensor with the same data, but different size
		x = x.view(-1, self.num_flat_features(x))      # returns number of features
		x = F.relu(self.fc1(x))                        # ReLU after first fully connected pass
		x = F.relu(self.fc2(x))                        # again
		x = self.fc3(x)                                # output everything
		return x


	def num_flat_features(self, x):
		'''
		- x is the current layer status in the network
		returns the number of available features at this point by flattening
		out the current 'layer'
		'''
		size = x.size()[1:]    # all dimensions except the batch dimensions?
		num_features = 1       # initialize counter
		for s in size:
			num_features *= s
		return num_features


''' -------------------------------------------------------- '''
'''   Modified Sample Network for working with color images  '''
''' -------------------------------------------------------- '''

class SimpleNet(nn.Module):
	'''
	- 32x32 pixel images
	- 3 channel images as input
	- 100 outputs (classes)
	'''

	def __init__(self):

		super(SimpleNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 110)
		self.fc3 = nn.Linear(110, 100)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# x = x.view(-1, 16 * 5 * 5)
		x = x.view(x.size(0), 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


''' -------------------------------------------------------- '''
'''      sample Alexnet network from PyTorch's examples      '''
''' -------------------------------------------------------- '''

class AlexNet(nn.Module):
	def __init__(self, num_classes=11):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x


''' -------------------------------------------------------- '''
'''          modified AlexNet for debugging purposes         '''
''' -------------------------------------------------------- '''

class incrementalAlexNet(nn.Module):
	def __init__(self, num_classes=11, batch_size=4):
		super(incrementalAlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.BatchNorm2d(64), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.BatchNorm2d(192), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.BatchNorm2d(384), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2)
			# nn.MaxPool2d(kernel_size=1, stride=2),			
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 3 * 3, 4096),
			# nn.Linear(256, 4096), # removed the 6x6, becuase our last layer just outputs a 256-sized vec.
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 3 * 3)
		# x = x.view(x.size(0), 256 ) # removed the 6x6, becuase our last layer just outputs a 256-sized vec.
		x = self.classifier(x)
		return x




''' -------------------------------------------------------- '''
'''          simplified AlexNet for comparison purposes      '''
''' -------------------------------------------------------- '''

class simplifiedAlexNet(nn.Module):
	def __init__(self, num_classes=11, batch_size=4):
		super(simplifiedAlexNet, self).__init__()
		self.features = nn.Sequential(
			# 1 conv
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			# 2 conv
			nn.Conv2d(64, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.classifier = nn.Sequential(
			# Fully 1
			nn.Linear(256 * 7 * 7, 4096),
			nn.ReLU(inplace=True),
			# Fully 2
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 7 * 7)
		x = self.classifier(x)
		return x




''' -------------------------------------------------------- '''
'''      simplified AlexNet with Dropout and BatchNorm       '''
''' -------------------------------------------------------- '''

class simpleDropBatchAlexNet(nn.Module):
	def __init__(self, num_classes=11, batch_size=4):
		super(simpleDropBatchAlexNet, self).__init__()
		self.features = nn.Sequential(
			# 1 conv
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.BatchNorm2d(64), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			# 2 conv
			nn.Conv2d(64, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256), #put in the number of features from the expected input			
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.classifier = nn.Sequential(
			# Fully 1
			nn.Dropout(),
			nn.Linear(256 * 7 * 7, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(inplace=True),
			# Fully 2
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 7 * 7)
		x = self.classifier(x)
		return x



''' -------------------------------------------------------- '''
'''          deeper AlexNet for debugging purposes           '''
''' -------------------------------------------------------- '''

class deepAlexNet(nn.Module):
	def __init__(self, num_classes=11, batch_size=4):
		super(deepAlexNet, self).__init__()
		self.features = nn.Sequential(
			# conv 1
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.BatchNorm2d(64), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			# conv 2
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.BatchNorm2d(192), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			# conv 3
			nn.Conv2d(192, 320, kernel_size=3, padding=1),
			nn.BatchNorm2d(320), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			# conv 4
			nn.Conv2d(320, 384, kernel_size=3, padding=1),
			nn.BatchNorm2d(384), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			# conv 5
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.BatchNorm2d(384), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			# conv 6
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			# conv 7
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256), #put in the number of features from the expected input
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.classifier = nn.Sequential(
			# linear 1
			nn.Dropout(),
			nn.Linear(256 * 3 * 3, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(inplace=True),
			# linear 2
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(inplace=True),
			# linear 3
			nn.Dropout(),
			nn.Linear(4096, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(inplace=True),
			# linear 4
			nn.Dropout(),
			nn.Linear(2048, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			# linear 5, output
			nn.Linear(1024, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 3 * 3)
		x = self.classifier(x)
		return x




