import sys
def heatmap_gen(Conv_Method,n_num,dset_name):
    from baselines import Plain,Rand,Randn,gat,gcn
    from syndata.SynDataset import SynDataset
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    import random
    import numpy as np
    import math
    import scipy.io as sio
    def train(train_mask):
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data)[train_mask], data.y[train_mask]).backward()
        optimizer.step()

    def test(train_mask,val_mask,test_mask):
        model.eval()
        logits, accs = model(data), []
        for mask in [train_mask, val_mask, test_mask]:
            pred = logits[mask].max(1)[1]
            #print(pred)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    dataset=SynDataset(root='data/'+dset_name+'_nnodes'+str(n_num),name=dset_name+'_nnodes'+str(n_num))
    times=range(10)
    pipeline_acc={Conv_Method:[i for i in times]}
    mn_score=[0]*len(dataset)
    indx=0
    sel_feats=10
    len_mul=int(n_num/5)
    best_acc=0
    for i in range(len(dataset)):
        data=dataset[i]
        data.x=data.x[:,:sel_feats]
        data.x=torch.ones(data.x.size())
        for time in times:
            index=[i for i in range(len(data.y))]
            random.shuffle(index)
            train_mask=torch.tensor([i < len_mul*2 for i in index])
            val_mask=torch.tensor([(i >= len_mul*2) and (i < len_mul*4) for i in index])
            test_mask=torch.tensor([i >= (len(data.y)-len_mul) for i in index])
            model,data = locals()[Conv_Method].call(data,dataset.name,data.x.size(1),dataset.num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            best_val_acc = test_acc = 0.0
            for epoch in range(1, 101):
                train(train_mask)
                train_acc,val_acc,tmp_test_acc = test(train_mask,val_mask,test_mask)
                if val_acc>=best_val_acc:
                    test_acc=tmp_test_acc
                    best_val_acc = val_acc
                pipeline_acc[Conv_Method][time]=test_acc
            log ='Epoch: 200, train acc: '+ str(train_acc) + ', val acc: '+ str(val_acc) + ' dataset:' + str(i)  + ' acc: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_Method][time])))
            if train_acc>best_acc:
                best_acc=train_acc
                torch.save(model.state_dict(),'bestmlp.pt')
                train_idx=train_mask.tolist()
                val_idx=val_mask.tolist()
                edge_idx=dataset[i].edge_index.tolist()
                ricci_cur=dataset[i].w_mul.tolist()
                graph_pred=model(data).max(1)[1]
                bestid=i
            del data
            del model
            torch.cuda.empty_cache()
            data=dataset[i]
            data.x=data.x[:,:sel_feats]
        torch.cuda.empty_cache()
        mn_score[indx]=np.mean(pipeline_acc[Conv_Method])
        indx+=1
    heat_map=np.reshape(mn_score,[int(math.sqrt(len(dataset))),int(math.sqrt(len(dataset)))])
    sio.savemat('scores/heat_map'+dset_name+ str(n_num)+Conv_Method+'.mat',{'heat_map':heat_map})
    graph_pred=graph_pred.tolist()
    sio.savemat('scores/graph'+dset_name+ str(n_num) +Conv_Method+'.mat',{'graph_pred':graph_pred,'edge_idx':edge_idx,'ricc_cur':ricci_cur,'train_idx':train_idx,'val_idx':val_idx,'bestid':bestid})

def main():
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    heatmap_gen(sys.argv[1],int(sys.argv[2]),sys.argv[3])

if __name__=='__main__':
    main()
