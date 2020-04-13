# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:01:56 2020

@author: Jake
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(20, 100)
        self.linear_2 = torch.nn.Linear(100,100)
        self.linear_3 = torch.nn.Linear(100,2)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.linear_2(x)
        x = self.selu(x)
        x = self.linear_3(x)
        pred = self.softmax(x)

        return pred
    
def hessian_vector_product(ys,xs,v):
    J = grad(ys,xs, create_graph=True)[0]
    J.backward(v,retain_graph=True)
    return xs.grad
    
r = np.random.RandomState(seed=1234567890)

x_imputed = np.load('data/x_imputed.npy')
y = np.load('data/y.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.is_available()):
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

mlp_probs = np.array([])
test_labels = np.array([])
test_data = np.array([])

no_epochs = 2000
train_loss = list()
val_loss = list()
best_val_loss = 1

sm = SMOTE()

x_train,x_test,y_train,y_test = train_test_split(x_imputed,y,test_size=0.1,random_state=r)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# case_idx = np.where(y_test==1)[0]
# x_test = x_test[case_idx]
# y_test = y_test[case_idx]
    
x_train, y_train = sm.fit_resample(x_train, y_train)

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

if device:
    model.to(device)
    print('Moved to GPU')

print_iter = 500

for epoch in range(no_epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    # training

    image = x_train
    label = y_train

    optimizer.zero_grad()
    
    pred = model(image)

    loss = criterion(pred, label)
    total_train_loss += loss.item()

    loss.backward()
    optimizer.step()

    train_loss.append(total_train_loss)

    # validation
    model.eval()
    total = 0

    image = x_test
    label = y_test
    
    pred_test = model(image)

    loss = criterion(pred_test, label)
    total_val_loss += loss.item()

    val_loss.append(total_val_loss)
    if(epoch % print_iter == 0 or epoch == no_epochs-1):
        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Test Loss: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss))

start_time = time.time()
model.zero_grad()
    
train_loss = criterion(model(x_train),y_train)
    
test_loss = criterion(model(x_test),y_test)

test_loss.backward(create_graph=True)

scale = 1000
damping = 1
num_samples = 1
recursion_depth=1000
print_iter = recursion_depth/10
v = model.linear_3.weight.grad.clone()
cur_estimate = v.clone()
for i in range(recursion_depth):
    hvp = hessian_vector_product(train_loss, model.linear_3.weight, cur_estimate)
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

# print(ihvp)

eqn_2 = np.array([])
eqn_5 = np.array([])

ihvp = ihvp.detach()
print_iter = int(len(x_train)/100)
print_cntr = 0
for i in range(len(x_train)):
    x = x_train[i]
    x.requires_grad = True
    x_out = model(x.view(1,-1))
    x_loss = criterion(x_out,y_train[i].reshape(1))
    x_loss.backward(create_graph=True)
    grads = model.linear_3.weight.grad
    grads = grads.squeeze()
    
    infl = (torch.dot(ihvp.view(1,-1).squeeze(),grads.view(-1,1).squeeze())/len(x_train))
    i_pert = grad(infl,x)
    i_pert = i_pert[0]

    eqn_2 = np.vstack((eqn_2,-infl.detach().cpu().numpy())) if eqn_2.size else -infl.detach().cpu().numpy()
    eqn_5 = np.vstack((eqn_5,-i_pert.detach().cpu().numpy())) if eqn_5.size else -i_pert.detach().cpu().numpy()
    model.zero_grad()
    if (i % print_iter ==0) or (i==len(x_train)-1):
        print("Done "+str(print_cntr)+"/100")
        print_cntr +=1
    
np.save('results/eqn_2-test_set_smote.npy',eqn_2)
np.save('results/eqn_5-test_set_smote.npy',eqn_5)

elapsed_time = time.time()-start_time
# np.save('results/mlp_influence_time',elapsed_time)