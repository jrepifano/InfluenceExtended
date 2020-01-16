# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:43:15 2019

@author: epifanoj0

Trian CNN to visualize attribution scores
"""
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear_1 = torch.nn.Linear(784, 100)
#         self.linear_2 = torch.nn.Linear(100,10)
#         self.selu = torch.nn.SELU()
#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         x = x.reshape(x.size(0), -1)
#         x = self.linear_1(x)
#         x = self.selu(x)
#         x = self.linear_2(x)
#         pred = self.softmax(x)

#         return pred

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=2, padding=1)
        self.linear_1 = torch.nn.Linear(4225, 10)
        # self.linear_2 = torch.nn.Linear(100,10)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.selu(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        # x = self.selu(x)
        # x = self.linear_2(x)
        pred = self.softmax(x)

        return pred
    
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

print("Training dataset size: ", len(mnist_trainset))
print("Validation dataset size: ", len(mnist_valset))
print("Testing dataset size: ", len(mnist_testset))


# torch.save(mnist_trainset,'mnist_trainset.pt')
# torch.save(mnist_testset,'mnist_testset.pt')
# torch.save(mnist_valset,'mnist_valset.pt')

# visualize data
# fig=plt.figure(figsize=(20, 10))
# for i in range(1, 6):
#     img = transforms.ToPILImage(mode='L')(mnist_trainset[i][0])
#     fig.add_subplot(1, 6, i)
#     plt.title(mnist_trainset[i][1])
#     plt.imshow(img)
# plt.show()


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

no_epochs = 50
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
    for itr, (image, label) in enumerate(train_dataloader):

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
    for itr, (image, label) in enumerate(val_dataloader):

        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = total / len(mnist_valset)

    total_val_loss = total_val_loss / (itr + 1)
    val_loss.append(total_val_loss)

    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))

fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_loss, label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()


torch.save(model,'cnn_2.pt')