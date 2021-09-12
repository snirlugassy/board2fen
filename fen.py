from torch import Tensor
from config import piece_idx, idx_piece


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
                _row.append(piece_idx[sym])
        output.append(_row)
    return Tensor(output).long()


def matrix2fen(m):
    assert m.shape == (8,8)
    label = ''
    for i in range(8):
        row = m[i]
        zeros = 0
        for j in row:
            if j != 0:
                if zeros != 0:
                    label += str(zeros)
                    zeros = 0
                label += str(idx_piece[int(j)])
            else:
                zeros += 1
        if zeros != 0:
            label += str(zeros)
        label += '-' if i<7 else ''
    return label
