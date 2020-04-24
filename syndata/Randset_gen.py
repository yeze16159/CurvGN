import numpy as np
import torch
from torch_geometric.data import Data
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
from SynDataset import SynDataset
#create random graph dataset

f_dim=100
for dset_i in [10]:
    c1,c2,c3,c4,c5=20*dset_i,40*dset_i,60*dset_i,80*dset_i,100*dset_i
    #c1,c2=500,1000
    n_num=c5
    p_list=[0.1*i for i in range(10)]
    adj=np.random.rand(n_num,n_num)
    adj=(adj+adj.T)/2
    x=torch.tensor(np.random.rand(n_num,f_dim),dtype=torch.float)
    #y=torch.cat((torch.tensor([0]*c1),torch.tensor([1]*(c2-c1))))
    y=torch.cat((torch.tensor([0]*c1),torch.tensor([1]*c1),torch.tensor([2]*c1),torch.tensor([3]*c1),torch.tensor([4]*c1)))
    end=n_num*n_num
    dataid=0
    inter_offset=0.95
    intra_offset=0.75
    prob_num=10
    inter_step=(1.0-inter_offset)/prob_num
    intra_step=(inter_offset-intra_offset)/prob_num
    data_list =[Data() for i in range(prob_num*prob_num)]
    for dataid in range(prob_num*prob_num):
        print(dataid)
        p_inter=inter_offset+(dataid%prob_num)*inter_step
        p_intra=intra_offset+(dataid/prob_num)*intra_step
        inter_idx=adj>p_inter
        intra_idx=adj>p_intra
        inter_idx[c4:,c3:c4],inter_idx[c3:c4,c4:]=False,False
        inter_idx[c1:c3,c3:c4],inter_idx[c3:c4,c1:c3]=False,False
        inter_idx[c1:c4,c4:],inter_idx[c4:,c1:c4]=False,False
        inter_idx[:c1,:c1]=intra_idx[:c1,:c1]
        inter_idx[c1:c2,c1:c2]=intra_idx[c1:c2,c1:c2]
        inter_idx[c2:c3,c2:c3]=intra_idx[c2:c3,c2:c3]
        inter_idx[c3:c4,c3:c4]=intra_idx[c3:c4,c3:c4]
        inter_idx[c4:c5,c4:c5]=intra_idx[c4:c5,c4:c5]
        adj_b=inter_idx
        adj_b.flat[:end:n_num+1]=False
        edge_index=[(i,j) for i in range(n_num) for j in range(n_num) if adj_b[i,j]]
        Gd=nx.Graph()
        Gd.add_edges_from(edge_index)
        Gd_OT=OllivierRicci(Gd, alpha=0.5, method="OTD", verbose="INFO")
        Gd = Gd_OT.compute_ricci_curvature()
        ricci_list=[]
        for n1,n2 in Gd.edges():
            ricci_list.append([n1,n2,Gd[n1][n2]['ricciCurvature']])
            ricci_list.append([n2,n1,Gd[n1][n2]['ricciCurvature']])
        ricci_list=sorted(ricci_list)
        w_mul=torch.tensor([i[2] for i in ricci_list])
        data=Data(x=x,edge_index=torch.tensor(edge_index).transpose(0,1),y=y)
        data.w_mul=torch.tensor(w_mul,dtype=torch.float)
        data_list[dataid]=data
    SynDataset(root='../data/Rand_nnodes'+str(n_num),name='Rand_nnodes'+str(n_num),data_list=data_list)
