import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import sys

def loaddatas(d_loader,d_name):
    if d_loader=='Planetoid':
        dataset = getattr(torch_geometric.datasets, d_loader)('../data/'+d_name,d_name,T.NormalizeFeatures())
    else:
        dataset = getattr(torch_geometric.datasets, d_loader)('../data/'+d_name,d_name)
    return dataset
