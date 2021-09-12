import random
import math
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F

from dataset import ChessPositionDataset
from transform import transformations
from model import ChessPositionNet
from fen import fen2matrix

from config import target_dim
from config import device
from config import validation_frac
from config import epochs
from config import state_file_name
from config import learning_rate

def run_validation(net, data, ids):
    correct = 0
    for sample_id in ids:
        _board, _labels = data[sample_id]
        for _i in range(8):
            for _j in range(8):
                _x,_y = _board[_i][_j], _labels[_i][_j]
                predicted = F.log_softmax(net(_x), dim=1).argmax()
                correct += int(_y == predicted)
    return correct / (len(ids) * 8 * 8)

print('-- TRAINING --')

# Datasets
train_data = ChessPositionDataset(img_dir='dataset/train', target_transform=fen2matrix)

# Dataloaders
train_dataloader = DataLoader(train_data, shuffle=True)

print(f'Loaded {len(train_data)} training samples')

print('Using device', device)
model = ChessPositionNet(target_dim=target_dim).to(device)

try:
    model.load_state_dict(torch.load(state_file_name, map_location=device))
except Exception:
    print('No saved state, model state is new')

optimizer = Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

# the number of samples to train in each epoch
total_samples = 2000

# use for training over all samples in each epoch
# total_samples = len(train_data)

train_ids = list(range(len(train_data)))
train_val_split_pos = math.floor(total_samples * validation_frac)

print(f'Training samples: {total_samples - train_val_split_pos}')
print(f'Validation samples: {train_val_split_pos}')

for _e in range(epochs):
    # randomly choose training and validation indices
    sample_ids = random.sample(train_ids, total_samples)
    validation_ids = sample_ids[:train_val_split_pos]
    train_ids = sample_ids[train_val_split_pos:]

    start_time = datetime.now().timestamp()
    epoch_loss = 0.0
    print(f'Epoch: {_e+1}/{epochs}')
    count = 0
    for k in train_ids:
        count += 1
        print(f'{count}/{len(train_ids)}',end='\r')
        board, labels = train_data[k]
        board_loss = 0.0
        for i in range(8):
            for j in range(8):
                y = model(board[i][j]).to(device)
                board_loss += loss(y, labels[i][j].reshape(1))

        # backpropagation
        optimizer.zero_grad()
        board_loss.backward()
        optimizer.step()
        
        epoch_loss += int(board_loss)

    epoch_duration = datetime.now().timestamp() - start_time

    print(f'Epoch took {epoch_duration:.2f} second')
    print(f'Epoch loss: {epoch_loss}')

    validation_acc = run_validation(model, train_data, validation_ids)
    print(f'Validation accuracy: {validation_acc}')

    # save model state to file
    torch.save(model.state_dict(), state_file_name)
