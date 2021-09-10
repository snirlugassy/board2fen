import torch
from config import fen_index


def fen2matrix(y):
    output = list()
    _y = y.split('-')
    for row in _y:
        _row = list()
        for sym in row:
            if sym.isdigit():
                for _i in range(int(sym)):
                    _row.append(0)
            else:
                _row.append(fen_index[sym])
        output.append(_row)
    return torch.tensor(output)
