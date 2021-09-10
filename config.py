import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
fen_index = {
    "q": 1,
    "k": 2,
    "p": 3,
    "n": 4,
    "b": 5,
    "r": 6,
    "Q": 7,
    "K": 8,
    "P": 9,
    "N": 10,
    "B": 11,
    "R": 12
}
target_dim = len(fen_index.keys()) + 1
channels = 3
board_size = 400
square_size = int(board_size / 8)
num_of_filters = 12
filter_size = 4
stride = 1
flat_layer_size = num_of_filters * (50-2*(filter_size-1))**2
hidden_dim = 50
dropout_rate = 0.1
