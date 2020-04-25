import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax,degree

class  curvGN(MessagePassing):
    def __init__(self, in_channels, out_channels,curv_n,w_mul,bias=True):
        super(curvGN, self).__init__(aggr='add') # "Add" aggregation.
        self.w_mul = w_mul
        self.lin=Linear(in_channels,out_channels)
        widths=[1,out_channels]
        self.w_mlp_out=create_wmlp(widths,out_channels,1)
    def forward(self, x, edge_index):
        x = self.lin(x)
        out_weight=self.w_mlp_out(self.w_mul)
        out_weight=softmax(out_weight,edge_index[0])
        return self.propagate(x=x,edge_index=edge_index,out_weight=out_weight)
    def message(self,x_j,edge_index,out_weight):
        return out_weight*x_j
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out

def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)


