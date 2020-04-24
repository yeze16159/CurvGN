import numpy as np
import torch
from torch_geometric.data import Data
from GraphRicciCurvature.FormanRicci import formanCurvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature
from GraphRicciCurvature.RicciFlow import compute_ricciFlow
import networkx as nx
import random
from wattsDataset import wattsDataset
#create random graph dataset

f_dim=100;
dset_len=100;
for dset_i in range(1,11):
    data_list=[Data() for i in range(dset_len)]
    n_num=100*dset_i;
    x=torch.tensor(np.random.rand(n_num,f_dim),dtype=torch.float)
    #create different graph base on graph density and rewired probability
    for dataid in range(dset_len):
        neighbornum = 2*np.log(n_num)+dataid%(np.sqrt(dset_len))
        rewired = np.floor(dataid/(np.sqrt(dset_len)))/20+0.05
        Gd = nx.watts_strogatz_graph(n=n_num,k=int(neighbornum),p=rewired)
        edge_index=sorted(Gd.edges())
        nodes=sorted(Gd.nodes())
        Gd = ricciCurvature(Gd)
        ricci_list=[]
        for n1,n2 in Gd.edges():
            ricci_list.append([n1,n2,Gd[n1][n2]['ricciCurvature']])
            ricci_list.append([n2,n1,Gd[n1][n2]['ricciCurvature']])
        ricci_list=sorted(ricci_list)
        w_mul=[i[2] for i in ricci_list]
        nx.set_node_attributes(Gd, 0, 'label')
        ncommunity = 4
        p = 0.5
        seeds = random.sample(Gd.nodes, ncommunity)
        for i, seed in enumerate(seeds):
            Gd.nodes[seed]['label'] = i+1
        infected = set(seeds)
        queue = list(Gd.edges(seeds))
        while queue:
            random.shuffle(queue)
            seed, neighbor = queue.pop()
            if neighbor in infected:
                continue
            if random.random() > p:
                Gd.nodes[neighbor]['label'] = Gd.nodes[seed]['label']
                infected.add(neighbor)
                queue.extend(list(Gd.edges(neighbor)))
        y=[Gd.nodes[i]['label'] for i in Gd.nodes()]   
        data=Data(x=x,edge_index=torch.tensor(edge_index).transpose(0,1),y=torch.tensor(y,dtype=torch.long))
        data.w_mul=torch.tensor(w_mul,dtype=torch.float)
        data_list[dataid]=data
    wattsDataset(root='../data/watts_nnodes'+str(n_num),name='watts_nnodes'+str(n_num),data_list=data_list)
