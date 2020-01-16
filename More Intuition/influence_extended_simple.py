# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:41:34 2019

@author: epifanoj0
"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()

        self.lin1 = nn.Linear(4,2)
        self.lin2 = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        out1 = self.sigmoid(self.lin1(x))
        out2 = self.sigmoid(self.lin2(out1))

        return out2

    def forward_with_stages(self,x):
        x0 = self.lin1(x)
        x1 = self.sigmoid(x0)
        x2 = self.lin2(x1)
        x3 = self.sigmoid(x2)

        return x0,x1,x2,x3

def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    J.backward(v,retain_graph=True)
    return xs.grad

model = torch.load('model.pt')
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

train_X = Variable(torch.Tensor(x_train).float())
test_X = Variable(torch.Tensor(x_test).float())
train_y = Variable(torch.Tensor(y_train).float()).view(-1,1)
test_y = Variable(torch.Tensor(y_test).float()).view(-1,1)

criterion = nn.MSELoss()
model.zero_grad()
train_output = model.forward(train_X)
train_loss = criterion(train_output, train_y)

#Influence start
x_t = torch.tensor(x_test).float()
y_t = torch.tensor(y_test).float()
xt0,xt1,xt2,test_output = model.forward_with_stages(x_t)
test_loss = criterion(test_output,y_t)
test_loss.backward(create_graph=True)

num_train_samples = len(x_train)

scale = 10
damping = 1
num_samples = 1
recursion_depth=1000
print_iter = recursion_depth/10
v = model.lin2.weight.grad.clone()
cur_estimate = v.clone()
for i in range(recursion_depth):
    hvp = hessian_vector_product(train_loss, model.lin2.weight, cur_estimate)
    cur_estimate = [a+(1-damping)*b-c/scale for (a,b,c) in zip(v,cur_estimate,hvp)]
    cur_estimate = torch.squeeze(torch.stack(cur_estimate)).view(1,-1)

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
for i in range(len(train_X)):
    x = torch.tensor(train_X[i],requires_grad=True)
    # model.zero_grad()
    x0,x1,x2,x_out = model.forward_with_stages(x)
    x_loss = criterion(x_out,train_y[0])
    x_loss.backward(create_graph=True)
    grads = model.lin2.weight.grad
    grads = grads.squeeze()
    
    infl = (torch.dot(ihvp.view(1,-1).squeeze(),grads.view(-1,1).squeeze())/num_train_samples)
    i_pert = grad(infl,x)

    eqn_2 = np.vstack((eqn_2,infl.detach().numpy())) if eqn_2.size else infl.detach().numpy()
    eqn_5 = np.vstack((eqn_5,i_pert[0].detach().numpy())) if eqn_5.size else i_pert[0].detach().numpy()

# np.save('results/eqn_2_extended_lin2.npy',eqn_2)
# np.save('results/eqn_5_extended_lin2.npy',eqn_5)