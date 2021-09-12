import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

from dataset import ChessPositionDataset
from transform import transformations
from model import ChessPositionNet
from fen import fen2matrix, matrix2fen

from config import target_dim
from config import device
from config import state_file_name

from collections import Counter

def hamming_distance(s1, s2):
    d = 0
    for i in range(min(len(s1), len(s2))):
        d+= s1[i] != s2[i]
    d += abs(len(s1)-len(s2))
    return d


def run_test():
    # Datasets
    test_data = ChessPositionDataset(img_dir='dataset/test', target_transform=fen2matrix)
    print(f'Testing over {len(test_data)} samples')

    print('Using device', device)
    model = ChessPositionNet(target_dim=target_dim).to(device)
    model.load_state_dict(torch.load(state_file_name, map_location=device))
    
    correct = 0
    count = 0
    distances = []
    for board, labels in test_data:
        fen = matrix2fen(labels)
        predicted = torch.zeros_like(labels)
        count += 1
        print(f'{count}/{len(test_data)}', end='\r')
        for i in range(8):
            for j in range(8):
                x,y = board[i][j], labels[i][j]
                _label = F.log_softmax(model(x), dim=1).argmax()
                predicted[i][j] = _label
                correct += int(y == _label)
        predicted_fen = matrix2fen(predicted)
        distances.append(hamming_distance(fen, predicted_fen))
    accuracy = correct / (count * 8 * 8)
    print(f'Test accuray: {100*accuracy:.2f}%')

    print(Counter(distances))
    # plt.hist(distances, bins=64)
    # plt.title('Output Hamming Distance Histogram')
    # plt.show()

    return accuracy

run_test()