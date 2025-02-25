import torch
import torch. nn as nn
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim, bias=False)

        torch.nn.init.trunc_normal_(self.layer_1.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)


        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.trunc_normal_(self.layer_2.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)


        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.trunc_normal_(self.layer_3.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)

        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.trunc_normal_(self.layer_4.weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)

        self.layer_5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))

        x = torch.relu(self.layer_3(x))

        x = torch.sigmoid(self.layer_4(x))

        x = self.layer_5(x)

        return x

def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()}, path)

def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).sum() / weight.sum()