from torch.nn import Module, Conv2d, Linear, LogSoftmax
from torch import flatten
from torch.nn.functional import relu

from config import num_of_filters
from config import filter_size
from config import stride
from config import flat_layer_size
from config import hidden_dim
from config import channels
from config import square_size


class ChessPositionNet(Module):
    def __init__(self, target_dim):
        super().__init__()     
        self.conv1 = Conv2d(channels, num_of_filters, kernel_size=filter_size, stride=stride)
        self.conv2 = Conv2d(num_of_filters, num_of_filters, kernel_size=filter_size, stride=stride)
        self.fc1 = Linear(flat_layer_size, hidden_dim)
        self.fc2 = Linear(hidden_dim, target_dim)

    def forward(self, x):
        """
        TODO:
        - Add dropout
        """

        x = x.reshape(1,channels,square_size,square_size)
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = relu(self.fc1(x))
        return self.fc2(x)
