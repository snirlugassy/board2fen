from torch import Tensor
from config import fen_index


def fen2matrix(y):
    output = []
    _y = y.split('-')
    for row in _y:
        _row = []
        for sym in row:
            if sym.isdigit():
                for _ in range(int(sym)):
                    _row.append(0)
            else:
                _row.append(fen_index[sym])
        output.append(_row)
    return Tensor(output).long()
