import random
import math

from dataset import ChessPositionDataset
from transform import transform
from model import ChessPositionNet
from fen import fen2matrix
from config import target_dim
from config import device

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F

def run_validation(model, data, ids):
    correct = 0
    for i in ids:
        board, labels = data[i]
        for _i in range(8):
            for _j in range(8):
                _x,_y = board[_i][_j], labels[_i][_j]
                predicted = F.log_softmax(model(_x), dim=0).argmax()
                correct += int(_y == predicted)
    return correct / (len(ids) * 8 * 8)

# Datasets
train_data = ChessPositionDataset(img_dir='dataset/train', transform=transform, target_transform=fen2matrix)
test_data = ChessPositionDataset(img_dir='dataset/train', transform=transform, target_transform=fen2matrix)

# Dataloaders
train_dataloader = DataLoader(train_data, shuffle=True)
test_dataloader = DataLoader(test_data, shuffle=True)

total_samples = len(train_data)
total_samples = 20
train_size = math.floor(0.7 * total_samples)
print(f'> Train size: {train_size}')

validation_size = total_samples - train_size
print(f'> Validation size: {validation_size}')

validation_ids = random.sample(range(total_samples), validation_size)
train_ids = [i for i in range(total_samples) if i not in validation_ids]

model = ChessPositionNet(target_dim=target_dim).to(device)

epochs = 10
optimizer = Adam(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

for _e in range(epochs):
    epoch_loss = 0
    print(f'> Epoch: {_e+1}/{epochs}')
    for k in train_ids:
        optimizer.zero_grad()
        board, labels = train_data[k]
        output = 0
        for i in range(8):
            for j in range(8):
                y = model(board[i][j]).to(device)
                output += loss(y, labels[i][j].reshape(1))
        output.backward()
        epoch_loss += int(output)
    
    print(f'Epoch loss: {epoch_loss}')
    
    validation_acc = run_validation(model, train_data, validation_ids)
    print(f'> Validation accuracy: {validation_acc}')
    
# TODO: save model parameters to file