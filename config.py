import torch

# the device used for storing and using tesnors
device = "cuda" if torch.cuda.is_available() else "cpu"

# the convertion from chess piece in FEN to label
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

# the number of possible target labels
target_dim = len(fen_index.keys()) + 1  

# number of input channel - RGB
channels = 3  

# the size of the chess board
board_size = 400  

# the size of 1 squre in the chess board
square_size = int(board_size / 8)  

# number of filters to apply / depth of the conv layer
num_of_filters = 12  

# convolution filter width and height
filter_size = 4  

# convolution filter stride
stride = 1  

# the size of the flat layer after the conv layers
flat_layer_size = num_of_filters * (50-2*(filter_size-1))**2 

# hidden layer dimension
hidden_dim = 50  

# the random channel dropout probability
dropout_rate = 0.1  

# the % of samples to take from training to validation
validation_frac = 0.3  

# number of training iterations
epochs = 10 