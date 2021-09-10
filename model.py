import torch
import torch.nn as nn
import torch.nn.functional as F

from config import num_of_filters
from config import filter_size
from config import stride
from config import flat_layer_size
from config import hidden_dim
from config import channels
from config import square_size


class ChessPositionNet(nn.Module):
    def __init__(self, target_dim):
        super().__init__()     
        self.conv1 = nn.Conv2d(channels, num_of_filters, kernel_size=filter_size, stride=stride)
        self.conv2 = nn.Conv2d(num_of_filters, num_of_filters, kernel_size=filter_size, stride=stride)
        self.fc1 = nn.Linear(flat_layer_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, target_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x)

    # def forward(self, board):
    #     """
    #     TODO:
    #     - Add dropout
    #     """

    #     targets = torch.zeros((8,8,13))
    #     assert board.shape[0] == 8
    #     assert board.shape[1] == 8
        
    #     for i in range(8):
    #         for j in range(8):
    #             x = board[i][j]
    #             x = x.reshape(1,3,50,50)
    #             x = F.relu(self.conv1(x))
    #             x = F.relu(self.conv2(x))
    #             x = torch.flatten(x, 1) # flatten all dimensions except batch
    #             x = F.relu(self.fc1(x))
    #             x = self.fc2(x)
    #             x = self.log_softmax(x)
    #             targets[i][j] = x
    #     return targets

    # For a single piece in the board

    def forward(self, x):
        """
        TODO:
        - Add dropout
        """

        x = x.reshape(1,channels,square_size,square_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # return self.log_softmax(self.fc2(x)).reshape(TARGET_DIM)
        return self.fc2(x)
