from torch import nn, cuda, device, FloatTensor, as_tensor, int64, optim, argmax, no_grad
from torch.cuda import is_available
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Fun
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import torch
cpu_gpu = device("cuda" if is_available() else "cpu")

""" for transform words and category numbers """

attribute_encoder = preprocessing.LabelEncoder()
attribute_encoder.fit(['Water', 'Fire', 'Earth', 'Light', 'Dark'])
race_encoder = preprocessing.LabelEncoder()
race_encoder.fit(['God', 'Human', 'Demon', 'Beast', 'Dragon', 'Elf', 'Machina', 'Material'])


""" read train.csv file """

train_filename = "./train.csv"
with open(train_filename, newline='') as csvfile:
    train_csv = csv.DictReader(csvfile)
    X = []
    num_feats = ['Hit Point', 'Attack Point', 'Recovery', 'Total']
    Y = []
    for row in train_csv:
        num = []
        for key in num_feats:
            num.append(int(row[key]))
        num.append(attribute_encoder.transform(([row['Attribute']])))
        X.append(num)
        Y.append(race_encoder.transform([row['Race']]))
            

""" read test.csv file """
answer = {"id":[], "Race":[]} # store the answers

# TODO: Load the test data.
test_filename = "./test.csv"
with open(test_filename, newline='') as csvfile:
    test_csv = csv.DictReader(csvfile)
    test_X = []
    num_feats = ['Hit Point', 'Attack Point', 'Recovery', 'Total']
    test_Y = []
    for row in test_csv:
        num = []
        for key in num_feats:
            num.append(int(row[key]))
        num.append(attribute_encoder.transform(([row['Attribute']])))
        test_X.append(num)
        answer['id'].append(int(row['id']))
        test_Y.append(race_encoder.transform(['Demon']))


""" Preparing Dataset """
class ToS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X)
        types = self.X[:,4]
        x_onehot = Fun.one_hot(types, num_classes=5)*100
        self.X = torch.cat((self.X[:,:4], x_onehot),1)
        self.Y = torch.tensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # TODO: Return Input and Target (X, Y) pair.
        return self.X[idx], self.Y[idx]

data_length = len(X)
train_data = ToS(X[:int(data_length*1)], Y[:int(data_length*1)])
train_loader = DataLoader(train_data, batch_size = 128, shuffle = True)
valid_data = ToS(X[int(data_length*0.8):], Y[int(data_length*0.8):])
valid_loader = DataLoader(train_data, batch_size = 128, shuffle = True)


# TODO: Generate test data and dataloader.
data_length = len(test_X)
test_data = ToS(test_X, test_Y)
test_loader = DataLoader(test_data, shuffle = False)
print(test_data[0])
print(len(test_data))


""" Define Model """
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # TODO: Define each layer of your model here
        # Use methods or classes from torch.nn
        self.f1 = nn.Linear(9, 512)
        self.d = nn.Dropout(p=0.6)
        self.relu = nn.LeakyReLU(0.1)
        self.out = nn.Linear(512, 8)
    def forward(self, x):
        # TODO: Define the forward pass
        y1 = self.relu(self.d(self.f1(x)))
        y2 = (self.out(y1))
        return y2

model = DNN().to(cpu_gpu)
""" Define Loss Function """
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=5e-4)
""" Training """
num_epochs = 5000
rate =1
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0
    valid_loss = 0
    history = {"besti": -1, "best_loss": 999}
    model.train()
    T =0
    F =0
    for step, (X, target) in enumerate(train_loader):
        optimizer.zero_grad()
        y = model(X.float().to(cpu_gpu))
        count = (y.argmax(1).to(cpu_gpu)==target.view(-1).to(cpu_gpu))
        #print(count)
        for p in count:
            if p:
                T+=1
            else:
                F+=1
        loss = criterion(y, target.view(-1).to(cpu_gpu, dtype=int64))
        loss.backward()
        epoch_loss += loss.data*rate
        optimizer.step()
    """ validation"""
    model.eval()
    with no_grad():
        # TODO: Validation part. Use the validation data
        for step, (X, target) in enumerate(valid_loader):
            optimizer.zero_grad()
            y = model(X.float().to(cpu_gpu))
            loss = criterion(y, target.view(-1).to(cpu_gpu, dtype=int64))
            valid_loss += loss.data*rate
            optimizer.step()
        
    print("epoch: %3d, training loss: %.3f"%(epoch, epoch_loss))
    print("epoch: %3d, validation loss: %.3f"%(epoch, valid_loss))
    if valid_loss < history['best_loss']:
        history['besti'] = step
        history['best_loss'] = valid_loss
    print('True:',T,", False:",F)

""" Testing """
model.eval()
answer['Race'] = []
with no_grad():
    # TODO: Testing part. Use the test data
    # Remember to fill in "answer"
    epoch_loss = 0
    valid_loss = 0
    for step, (X, target) in enumerate(test_loader):
        y = model(X.float().to(cpu_gpu))
        out = race_encoder.inverse_transform([y.argmax().tolist()])[0]
        print(out)
        answer['Race'].append(out)
df = pd.DataFrame(answer)
df = df[["id", "Race"]]
df.to_csv("./submission.csv", index=False)




