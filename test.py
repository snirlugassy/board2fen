import torch
from torch.nn import functional as F

from dataset import ChessPositionDataset
from transform import transformations
from model import ChessPositionNet
from fen import fen2matrix

from config import target_dim
from config import device
from config import state_file_name

def run_test():
    # Datasets
    test_data = ChessPositionDataset(img_dir='dataset/test', transform=transformations, target_transform=fen2matrix)
    print(f'Testing over {len(test_data)} samples')

    print('Using device', device)
    model = ChessPositionNet(target_dim=target_dim).to(device)
    model.load_state_dict(torch.load(state_file_name, map_location=device))
    
    correct = 0
    count = 0
    for board, labels in test_data:
        count += 1
        print(f'{count}/{len(test_data)}', end='\r')
        for i in range(8):
            for j in range(8):
                x,y = board[i][j], labels[i][j]
                predicted = F.log_softmax(model(x), dim=1).argmax()
                correct += int(y == predicted)
    accuracy = correct / (count * 8 * 8)
    print(f'Test accuray: {100*accuracy:.2f}%')

    return accuracy

run_test()