# -*- coding: utf-8 -*-
"""
@author: epifanoj0
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch

x = np.load('data/data.npy')
y = np.load('data/labels.npy')
# feat_names = np.load('data/feat_names.npy') save in another format and fix this


x_train,x_test,y_train,y_test = train_test_split(x[:,:27],y,test_size=0.2)
# clf = LogisticRegression(solver='lbfgs').fit(x_train,y_train)

# y_test_pred = clf.predict(x_test)
# print(confusion_matrix(y_test,y_test_pred))

# print(clf.score(x_test,y_test))

# np.save('results/logreg_coefs.npy',clf.coef_)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(27, 100)
        self.linear_2 = torch.nn.Linear(100,1)
        self.selu = torch.nn.SELU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.linear_2(x)
        pred = self.sigmoid(x)

        return pred

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

model = Model()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

no_epochs = 100
train_loss = list()
val_loss = list()
best_val_loss = 1

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

for epoch in range(no_epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    # training

    image = (x_train).float()
    label = y_train.float()

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

    image = (x_test).float()
    label = y_test.float()
    
    pred = model(image)

    loss = criterion(pred, label)
    total_val_loss += loss.item()

    val_loss.append(total_val_loss)

    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Test Loss: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss))

fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_loss, label="Test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()


torch.save(model,'mlp.pt')