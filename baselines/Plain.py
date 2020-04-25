import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot,zeros
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax

class  plain(MessagePassing):
    def __init__(self, in_channels, out_channels,bias=True):
        super(plain, self).__init__(aggr='add') # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin=Linear(in_channels,out_channels)
    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(x=x,edge_index=edge_index)
    def message(self,x_j,edge_index):
        return x_j
    def update(self, aggr_out):
        return aggr_out

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes):
        super(Net, self).__init__()
        self.conv1 = plain(num_features,64)
        self.conv2 = plain(64, num_classes)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def call(data,name,num_features,num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    data = data.to(device)
    model= Net(data,num_features,num_classes).to(device)
    return model, data

