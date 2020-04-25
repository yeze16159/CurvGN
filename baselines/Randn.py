import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax
from baselines.curvGN import curvGN

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes):
        super(Net, self).__init__()
        self.conv1 = curvGN(num_features,64,64,data.w_mul)
        self.conv2 = curvGN(64, num_classes,num_classes,data.w_mul)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def call(data,name,num_features,num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    data.w_mul=torch.cat((data.w_mul,torch.tensor([0.0]*len(data.y))),dim=-1)
    #data.w_mul=softmax(data.w_mul.view(-1,1).squeeze(),data.edge_index[0].squeeze(),data.x.size(0))
    data.w_mul=(data.w_mul).exp()
    data.w_mul=data.w_mul.view(-1,1)
    data=data.to(device)
    model= Net(data,num_features,num_classes).to(device)
    return model, data

