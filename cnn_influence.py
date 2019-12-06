# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:28:18 2019

@author: epifanoj0

Influence of CNN
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear_1 = torch.nn.Linear(25088, 10)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.selu(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        pred = self.softmax(x)

        return pred

def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    J.backward(v,retain_graph=True)
    return xs.grad

model = torch.load('cnn.pt')

mnist_valset = torch.load('mnist_valset.pt')
mnist_testset = torch.load('mnist_testset.pt')
mnist_trainset = torch.load('mnist_trainset.pt')

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=60000, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=9000, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=1000, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
model.zero_grad()

for itr, (image, label) in enumerate(train_dataloader):
    train_output = model.forward(image)
    train_loss = criterion(train_output, label)


for itr, (test_images, test_labels) in enumerate(test_dataloader):
    test_output = model.forward(image)
    test_loss = criterion(test_output, test_labels)

test_loss.backward(create_graph=True)

num_train_samples = 60000

scale = 10
damping = 1
num_samples = 1
recursion_depth=1000
print_iter = recursion_depth/10
v = model.linear_1.weight.grad.clone()
cur_estimate = v.clone()
for i in range(recursion_depth):
    hvp = hessian_vector_product(train_loss, model.lin1.weight, cur_estimate)
    cur_estimate = [a+(1-damping)*b-c/scale for (a,b,c) in zip(v,cur_estimate,hvp)]
    cur_estimate = torch.squeeze(torch.stack(cur_estimate))#.view(1,-1)

    if (i % print_iter ==0) or (i==recursion_depth-1):
        numpy_est = cur_estimate[0].detach().numpy()
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
for i in range(60000):
    x = torch.tensor(image[0][i],requires_grad=True)
    # model.zero_grad()
    x0,x1,x2,x_out = model.forward_with_stages(x)
    x_loss = criterion(x_out,label[0])
    x_loss.backward(create_graph=True)
    grads = model.linear_1.weight.grad
    grads = grads.squeeze()
    
    infl = (torch.dot(ihvp.view(1,-1).squeeze(),grads.view(-1,1).squeeze())/num_train_samples)
    i_pert = grad(infl,x)

    eqn_2 = np.vstack((eqn_2,infl.detach().numpy())) if eqn_2.size else infl.detach().numpy()
    eqn_5 = np.vstack((eqn_5,i_pert[0].detach().numpy())) if eqn_5.size else i_pert[0].detach().numpy()

