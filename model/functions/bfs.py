import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import os
import sys

# Add current directory to path to find tree_filter_cuda.so
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import tree_filter_cuda as _C

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child =\
                _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child

bfs = _BFS.apply
