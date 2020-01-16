# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:01:27 2020

@author: Jake
"""

import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.linear_1 = torch.nn.Linear(25088, 100)
#         # self.linear_2 = torch.nn.Linear(100,10)
#         self.selu = torch.nn.SELU()
#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.selu(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.linear_1(x)
#         # x = self.selu(x)
#         # x = self.linear_2(x)
#         pred = self.softmax(x)

#         return pred

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(784, 100)
        self.linear_2 = torch.nn.Linear(100,10)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.linear_2(x)
        pred = self.softmax(x)

        return pred

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=2, padding=1)
#         self.linear_1 = torch.nn.Linear(4225, 10)
#         # self.linear_2 = torch.nn.Linear(100,10)
#         self.selu = torch.nn.SELU()
#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.selu(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.linear_1(x)
#         # x = self.selu(x)
#         # x = self.linear_2(x)
#         pred = self.softmax(x)

#         return pred
    
def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    J.backward(v,retain_graph=True)
    return xs.grad

model_type = 'mlp'
model = torch.load(model_type +'.pt')

mnist_trainset = torch.load('mnist_trainset.pt')
mnist_testset = torch.load('mnist_testset.pt')
mnist_valset = torch.load('mnist_valset.pt')

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=5000, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=1000, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

criterion = torch.nn.CrossEntropyLoss()
model.to(device)

# test_indices = mnist_testset.indices
# test_data = mnist_trainset.data[test_indices]
# test_labels = mnist_trainset.targets[test_indices]

test_index = 6

for itr, (image, label) in enumerate(train_dataloader):
    image = image.to(device)
    label = label.to(device)
    
    train_output = model(image)
    train_loss = criterion(train_output,label)
    break


for itr, (test_images, test_labels) in enumerate(test_dataloader):
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    break    

# test_image = test_images[test_index]
# test_label = test_labels[test_index]

# print('Finding Influence on test point with label: '+str(test_label.cpu().numpy()))

# test_loss = criterion(model(test_image.unsqueeze(0)),test_label.reshape(1))

print('Finding Influence on test point test set')

test_loss = criterion(model(test_images),test_labels)

test_loss.backward(create_graph=True)

scale = 10
damping = 1
num_samples = 1
recursion_depth=500
print_iter = recursion_depth/10
v = model.linear_1.weight.grad.clone()
cur_estimate = v.clone()
for i in range(recursion_depth):
    hvp = hessian_vector_product(train_loss, model.linear_1.weight, cur_estimate)
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
    grads = model.linear_1.weight.grad
    grads = grads.squeeze()
    
    infl = (torch.dot(ihvp.view(1,-1).squeeze(),grads.view(-1,1).squeeze())/len(image))
    i_pert = grad(infl,x)
    i_pert = i_pert[0].view(1,28,28)

    eqn_2 = np.vstack((eqn_2,-infl.detach().cpu().numpy())) if eqn_2.size else -infl.detach().cpu().numpy()
    eqn_5 = np.vstack((eqn_5,-i_pert.detach().cpu().numpy())) if eqn_5.size else -i_pert.detach().cpu().numpy()
    
    
sort = np.argsort(eqn_2.reshape(-1))

np.save('results/train_images.npy',image.detach().cpu().numpy())
np.save('results/train_labels.npy',label.detach().cpu().numpy())
np.save('results/'+model_type+'-eqn_2-test_set.npy',eqn_2)
np.save('results/'+model_type+'-eqn_5-test_set.npy',eqn_5)