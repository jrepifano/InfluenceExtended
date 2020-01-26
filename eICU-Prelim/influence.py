# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:07:33 2020

@author: Jake
"""

import torch
import numpy as np
from torch.autograd import grad

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(27, 100)
        self.linear_2 = torch.nn.Linear(100,2)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.linear_2(x)
        pred = self.softmax(x)

        return pred

    
def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    J.backward(v,retain_graph=True)
    return xs.grad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

x_train = np.load('data/x_train.npy')
x_test = np.load('data/x_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

model = torch.load('mlp.pt')

if device:
    model.to(device)
    print('Moved to GPU')
    
criterion = torch.nn.CrossEntropyLoss()

train_loss = criterion(model(x_train),y_train)
    
test_loss = criterion(model(x_test),y_test)

test_loss.backward(create_graph=True)

scale = 1000
damping = 1
num_samples = 1
recursion_depth=100
print_iter = recursion_depth/10
v = model.linear_2.weight.grad.clone()
cur_estimate = v.clone()
for i in range(recursion_depth):
    hvp = hessian_vector_product(train_loss, model.linear_2.weight, cur_estimate)
    cur_estimate = [a+(1-damping)*b-c/scale for (a,b,c) in zip(v,cur_estimate,hvp)]
    cur_estimate = torch.squeeze(torch.stack(cur_estimate))#.view(1,-1)
    model.zero_grad()
    if (i % print_iter ==0) or (i==recursion_depth-1):
        numpy_est = cur_estimate.detach().cpu().numpy()
        numpy_est = numpy_est.reshape(1,-1)
        print("Recursion at depth %s: norm is %.8lf" % (i,np.linalg.norm(np.concatenate(numpy_est))))
    ihvp = [b/scale for b in cur_estimate]
    ihvp = torch.squeeze(torch.stack(ihvp))
    ihvp = [a/num_samples for a in ihvp]
    ihvp = torch.squeeze(torch.stack(ihvp))

print(ihvp)

eqn_2 = np.array([])
eqn_5 = np.array([])

ihvp = ihvp.detach()
for i in range(len(x_train)):
    x = x_train[i]
    x.requires_grad = True
    x_out = model(x.view(1,-1))
    x_loss = criterion(x_out,y_train[i].reshape(1))
    x_loss.backward(create_graph=True)
    grads = model.linear_2.weight.grad
    grads = grads.squeeze()
    
    infl = (torch.dot(ihvp.view(1,-1).squeeze(),grads.view(-1,1).squeeze())/len(x_train))
    i_pert = grad(infl,x)
    i_pert = i_pert[0]

    eqn_2 = np.vstack((eqn_2,-infl.detach().cpu().numpy())) if eqn_2.size else -infl.detach().cpu().numpy()
    eqn_5 = np.vstack((eqn_5,-i_pert.detach().cpu().numpy())) if eqn_5.size else -i_pert.detach().cpu().numpy()
    model.zero_grad()
    
np.save('results/eqn_2-test_set.npy',eqn_2)
np.save('results/eqn_5-test_set.npy',eqn_5)