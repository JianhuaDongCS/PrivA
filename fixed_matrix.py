
import torch
import torch.nn as nn


def fixed_matrix_Age():
    """
    0-1:[1, 1, 0, 0, 0, 0, 0]
    2-3:[0, 0, 1, 1, 0, 0, 0]
    4-5:[0, 0, 0, 0, 1, 1, 0]
    6+ :[0, 0, 0, 0, 0, 0, 1]
    """
    fixed_matrix = torch.Tensor(     [[1, 1, 0, 0, 0, 0, 0]
                                    , [0, 0, 1, 1, 0, 0, 0]
                                    , [0, 0, 0, 0, 1, 1, 0]
                                    , [0, 0, 0, 0, 0, 0, 1]])
    return fixed_matrix

