# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:38:12 2020

@author: Jake
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad

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
    
def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    J.backward(v,retain_graph=True)
    return xs.grad

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
testloader = torch.utils.data.DataLoader(testset, batch_size=int(0.5*len(testset)),
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = torch.load('cnn.pt')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if device:
    model.to(device)
    print('Moved to GPU')
    

for itr, (image, label) in enumerate(trainloader):
    image = image.to(device)
    label = label.to(device)
    
    train_output = model(image)
    train_loss = criterion(train_output,label)
    break


for itr, (test_images, test_labels) in enumerate(testloader):
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    break    

print('Finding Influence on test point test set')

test_loss = criterion(model(test_images),test_labels)

test_loss.backward(create_graph=True)

scale = 10
damping = 1
num_samples = 1
recursion_depth=500
print_iter = recursion_depth/10
v = model.fc2.weight.grad.clone()
cur_estimate = v.clone()
for i in range(recursion_depth):
    hvp = hessian_vector_product(train_loss, model.fc2.weight, cur_estimate)
    cur_estimate = [a+(1-damping)*b-c/scale for (a,b,c) in zip(v,cur_estimate,hvp)]
    cur_estimate = torch.squeeze(torch.stack(cur_estimate))#.view(1,-1)
    model.zero_grad()
    if (i % print_iter ==0) or (i==recursion_depth-1):
        numpy_est = cur_estimate.detach().cpu().numpy()
        numpy_est = numpy_est.reshape(1,-1)####
        print("Recursion at depth %s: norm is %.8lf" % (i,np.linalg.norm(np.concatenate(numpy_est))))
    ihvp = [b/scale for b in cur_estimate]
    ihvp = torch.squeeze(torch.stack(ihvp))
    ihvp = [a/num_samples for a in ihvp]
    ihvp = torch.squeeze(torch.stack(ihvp))

print(ihvp)

eqn_2 = np.array([])
eqn_5 = np.array([])

ihvp = ihvp.detach()
for i in range(len(image)):
    model.zero_grad()
    x = image[i].unsqueeze(0)
    x.requires_grad = True
    x_out = model(x)
    x_loss = criterion(x_out,label[i].reshape(1))
    x_loss.backward(create_graph=True)
    grads = model.fc2.weight.grad
    grads = grads.squeeze()
    
    infl = (torch.dot(ihvp.view(1,-1).squeeze(),grads.view(-1,1).squeeze())/len(image))
    i_pert = grad(infl,x)
    i_pert = i_pert[0].view(1,3,32,32)

    eqn_2 = np.vstack((eqn_2,-infl.detach().cpu().numpy())) if eqn_2.size else -infl.detach().cpu().numpy()
    eqn_5 = np.vstack((eqn_5,-i_pert.detach().cpu().numpy())) if eqn_5.size else -i_pert.detach().cpu().numpy()
    
    
sort = np.argsort(eqn_2.reshape(-1))

np.save('results/train_images.npy',image.detach().cpu().numpy())
np.save('results/train_labels.npy',label.detach().cpu().numpy())
np.save('results/eqn_2-test_set.npy',eqn_2)
np.save('results/eqn_5-test_set.npy',eqn_5)