# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 00:13:14 2020

@author: Jake
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64,64,3)
        self.conv5 = nn.Conv2d(64,128,3)
        self.conv6 = nn.Conv2d(128,128,3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.selu = nn.SELU()
    def forward(self, x):
        x = self.selu(self.conv1(x))
        x = self.selu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.selu(self.conv3(x))
        x = self.selu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.selu(self.conv5(x))
        x = self.selu(self.conv6(x))
        # x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.selu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x
    
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(0.1*len(trainset)),shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=int(0.2*len(testset)),
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

no_epochs = 100
train_loss = list()
val_loss = list()
best_val_loss = 1

if device:
    model.to(device)
    print('Moved to GPU')
    

for epoch in range(no_epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    # training
    for itr, (image, label) in enumerate(trainloader):

        image =  image.to(device)
        label = label.to(device)
    
        optimizer.zero_grad()
        
        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss / (itr + 1)
    train_loss.append(total_train_loss)

    # validation
    model.eval()
    total = 0
    for itr, (image, label) in enumerate(testloader):

        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = total / len(testset)

    total_val_loss = total_val_loss / (itr + 1)
    val_loss.append(total_val_loss)

    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Test Loss: {:.8f}, Test Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))

torch.save(model,'cnn.pt')