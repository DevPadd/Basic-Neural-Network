import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd


# * Workflow
# input layer (features of flowers) ---> 
# hidden layer1 (number of neurons) ---> 
# H2 (n) --->
# output layer (3 classes of iris flowers)

# create a model class that inherits the nn.Module
class Model(nn.Module):
    def __init__(self,input_features=4, h1=8, h2=10,output_features=3):
        super().__init__()

        # these 3 lines of code declares 3 layers of the neural networks
        self.fc1 = nn.Linear(input_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,output_features)

    # 
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# picking manual seed for randomization
torch.manual_seed(67)

# create an instance of our model
model = Model()

# load our dataset
url = 'Iris.csv'
df = pd.read_csv(url)

print(df.head())


# todos: fix this abomination

# iris setosa = 0, iris versicolor = 1, iris virginica = 2
# basically changing all the str to int so our model can work with it easily
# df["Species"] = df['Species'].replace({'Iris-setosa':0,'Iris-versicolor':1,
#                 'Iris-virginica':2})


# # Train, test, and split. set x and y
# x = df.drop('Species',axis=1).values
# y = df['Species'].values

# # convert to tensors
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)