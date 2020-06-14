import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax
from baselines.curvGN import curvGN

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes,w_mul):
        super(Net, self).__init__()
        self.conv1 = curvGN(num_features,64,64,w_mul)
        self.conv2 = curvGN(64, num_classes,num_classes,w_mul)
    def forward(self,data):
        x = F.dropout(data.x,p=0.6,training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes):
    filename='../data/curvature/graph_'+name+'.edge_list'
    f=open(filename)
    cur_list=list(f)
    if name=='Cora' or name=='CS':
        ricci_cur=[[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
    else:
        ricci_cur=[[] for i in range(2*len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
            ricci_cur[i+len(cur_list)]=[ricci_cur[i][1],ricci_cur[i][0],ricci_cur[i][2]]
    ricci_cur=sorted(ricci_cur)
    w_mul=[i[2] for i in ricci_cur]
    #w_mul=[(i[2]+1)/2 for i in ricci_cur]
    w_mul=w_mul+[0 for i in range(data.x.size(0))]
    w_mul=torch.tensor(w_mul, dtype=torch.float)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    data.w_mul=w_mul.view(-1,1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #change to call function
    data.w_mul = data.w_mul.to(device)
    model, data = Net(data,num_features,num_classes,data.w_mul).to(device), data.to(device)
    return model, data
