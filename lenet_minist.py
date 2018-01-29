import torch  
import torch.nn as nn  
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision  
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.1
DOWNLOAD_MNIST = True
loss_fn = nn.CrossEntropyLoss()

def get_data(train=True):
    # get MNIST dataset  
    training_data = torchvision.datasets.MNIST(
                 root='mnist/',
                 train=train,
                 transform=torchvision.transforms.ToTensor(), # Normalize data to [0,1]
                 download=DOWNLOAD_MNIST,
                 )
    if train == True:
        data = training_data.train_data.numpy()
        label = training_data.train_labels.numpy()
    else:
        data = training_data.test_data.numpy()
        label = training_data.test_labels.numpy()       
    return data, label

def show_image(data, label):
    plt.imshow(data, cmap='gray')
    plt.title('%i' % label)
    plt.show()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        '''This line is different from pytorch Tutorial becasue the image size
        is 28 * 28'''
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_epoch(model, data, label, optimizer):

    def train_batch(model, batch_data, batch_label, optimizer):
        '''Step 1. Remember that Pytorch accumulates gradients.
        We need to clear them out before each instance'''
        model.zero_grad()
        
        '''Step 2. Get our inputs ready for the network, that is,
        turn them into Variables of word indices'''
        batch_data = Variable(torch.FloatTensor(np.array(batch_data).reshape(-1, 1, 28, 28)/255.0))
        targets = Variable(torch.LongTensor(batch_label))
        
        '''Step 3. Run our forward pass'''
        logic = model(batch_data)
        
        '''Step 4. Compute the loss, gradients,
        and update the parameters'''
        loss = loss_fn(logic, targets)
        loss.backward()
        optimizer.step()

    model.train()
    batch_data, batch_label = [], []
    for i, index in enumerate(np.random.permutation(len(data))):
        # Prepare batch dataset
        batch_data.append(data[index])
        batch_label.append(label[index])
        if len(batch_data) == BATCH_SIZE:
            # Train the model
            train_batch(model, batch_data, batch_label, optimizer)
            # Clear old batch
            batch_data, batch_label = [], []

def eval_epoch(model, data, label):
    model.eval()
    loss = 0.0
    for d, l in zip(data, label):
        single_data = Variable(torch.FloatTensor(d.reshape(-1, 1, 28, 28)/255.0))
        single_target = Variable(torch.LongTensor(l.reshape(-1,)))
        single_logic = model(single_data)
        single_loss = loss_fn(single_logic, single_target)
        loss += single_loss.data.numpy()
    return loss/len(data)

def main():
    train_data, train_label = get_data(train=True)
    test_data, test_label = get_data(train=False)
    print('The type of train data is %s, the shape is %s'%(str(type(train_data)), str(train_data.shape)))
    print('The type of train label is %s, the shape is %s'%(str(type(train_label)), str(train_label.shape)))
    print('The type of train data is %s, the shape is %s'%(str(type(test_data)), str(test_data.shape)))
    print('The type of train label is %s, the shape is %s'%(str(type(test_label)), str(test_label.shape)))


    model = Net()
    optimizer= optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = eval_epoch(model, test_data, test_label)
    print('The initial loss is: %f'%(loss,))
    for i in range(EPOCH):
        train_epoch(model, train_data, train_label, optimizer)
        loss = eval_epoch(model, test_data, test_label)
        print('EPOCH %d, loss %f'%(i, loss))

if __name__ == '__main__':    
    main()