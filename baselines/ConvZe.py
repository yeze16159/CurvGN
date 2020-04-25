import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
import numpy
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax
class  ConvZe(MessagePassing):
    def __init__(self, in_channels, out_channels,w_mul,bias=True):
        super(ConvZe, self).__init__(aggr='add') # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_mul = w_mul
        self.lin=Linear(in_channels,out_channels)
    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index,num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, x=x,num_nodes=x.size(0))
    def message(self,x_j, edge_index,num_nodes):
        out_weight=softmax(self.w_mul.view(-1,1),edge_index[0],num_nodes)
        return out_weight*x_j
    def update(self, aggr_out):
        return aggr_out

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes,w_mul):
        super(Net, self).__init__()
        self.conv1 = ConvZe(num_features,64,w_mul)
        self.conv2 = ConvZe(64, num_classes,w_mul)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=0.8,training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=0.4,training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes):
    load_cur=1
    if not load_cur:
        Gd=nx.DiGraph()
        edge_index = data.edge_index
        Gd.add_edges_from(edge_index.transpose(0,1).numpy())
        Gd = formanCurvature(Gd)
        w_mul=[Gd[n1][n2]['formanCurvature'] for n1,n2 in Gd.edges()]
    else:
        filename='../data/curvature/graph_'+name+'.edge_list'
        f=open(filename)
        cur_list=list(f)
        ricci_cur=[[] for i in range(len(cur_list)*2)]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
            ricci_cur[i+len(cur_list)]=[ricci_cur[i][1],ricci_cur[i][0],ricci_cur[i][2]]
        ricci_cur=sorted(ricci_cur)
        w_mul=[i[2] for i in ricci_cur]
    w_mul=w_mul+[0 for i in range(data.x.size(0))]
    w_mul=torch.tensor(w_mul, dtype=torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #change to call function
    w_mul = w_mul.to(device)
    model, data = Net(data,num_features,num_classes,w_mul).to(device), data.to(device)
    return model, data
