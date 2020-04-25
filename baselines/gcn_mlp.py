import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot,zeros
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax

class  RCGN(MessagePassing):
    def __init__(self, in_channels, out_channels,w_mul,bias=True):
        super(RCGN, self).__init__(aggr='add') # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_mul = w_mul
        self.lin=Linear(in_channels,out_channels)
        widths=[1,out_channels]
        self.w_mlp_out=create_wmlp(widths,out_channels,1)
    def forward(self, x, edge_index):
        x = self.lin(x)
        #out_weight=self.w_mul.view(-1,1)
        row, col = edge_index
        deg = degree(row,x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm =torch.tensor(norm).view(-1,1)
        norm.require_grads=True
        out_weight=self.w_mlp_out(norm)
        return self.propagate(x=x,edge_index=edge_index,out_weight=out_weight)
    def message(self,x_j,edge_index,out_weight):
        return out_weight*x_j
    def update(self, aggr_out):
        return aggr_out
    
def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)


class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes):
        super(Net, self).__init__()
        self.conv1 = RCGN(num_features,8,data.w_mul)
        self.conv2 = RCGN(8, num_classes,data.w_mul)
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
    data.edge_index = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    data=data.to(device)
    model= Net(data,num_features,num_classes).to(device)
    return model, data

