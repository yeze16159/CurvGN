import numpy as np
import torch
from torch_geometric.data import Data
from GraphRicciCurvature.FormanRicci import formanCurvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature
from GraphRicciCurvature.RicciFlow import compute_ricciFlow
import networkx as nx
import random
from SynDataset import SynDataset
#create random graph dataset

f_dim=100;
dset_len=100;
for dset_i in [5]:
    data_list=[Data() for i in range(dset_len)]
    n_num=100*dset_i;
    c=[20*dset_i,40*dset_i,60*dset_i,80*dset_i,100*dset_i];
    x=torch.tensor(np.random.rand(n_num,f_dim),dtype=torch.float)
    y=torch.cat((torch.tensor([0]*c[0]),torch.tensor([1]*(c[1]-c[0])),torch.tensor([2]*(c[2]-c[1])),torch.tensor([3]*(c[3]-c[2])),torch.tensor([4]*(c[4]-c[3]))))
    #create different graph base on graph density and rewired probability
    for dataid in range(dset_len):
        neighbornum = int(dset_i+dset_i*(dataid%(np.sqrt(dset_len))))
        rewired = np.floor(dataid/(np.sqrt(dset_len)))/50+0.02
        Gd = nx.newman_watts_strogatz_graph(n=n_num,k=int(neighbornum),p=rewired)
        for c_idx in range(len(c)-1):
            for node in range(c[c_idx]-neighbornum,c[c_idx]):
                remove_list=[ npair for npair in Gd.edges(node) if npair[1]>c[c_idx]-1 and npair[1]-npair[0]<neighbornum]
                Gd.remove_edges_from(remove_list)
        for node in range(c[-1]-neighbornum,c[-1]):
            remove_list=[npair for npair in Gd.edges(node) if npair[0]-npair[1]>neighbornum and npair[1]+100-npair[0]<neighbornum]
            Gd.remove_edges_from(remove_list)
        edge_index=sorted(Gd.edges())
        nodes=sorted(Gd.nodes())
        Gd = ricciCurvature(Gd)
        ricci_list=[]
        for n1,n2 in Gd.edges():
            ricci_list.append([n1,n2,Gd[n1][n2]['ricciCurvature']])
            ricci_list.append([n2,n1,Gd[n1][n2]['ricciCurvature']])
        ricci_list=sorted(ricci_list)
        w_mul=[i[2] for i in ricci_list]
        data=Data(x=x,edge_index=torch.tensor(edge_index).transpose(0,1),y=y)
        data.w_mul=torch.tensor(w_mul,dtype=torch.float)
        data_list[dataid]=data
    SynDataset(root='../data/Nwatts_nnodes'+str(n_num),name='Nwatts_nnodes'+str(n_num),data_list=data_list)
