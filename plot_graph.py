import sys
def plot_graph(Conv_Method,n_num):
    from baselines import Rand,Plain,Randn,gat,gcn,gcn_mlp
    from syndata.SynDataset import SynDataset
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    import random
    import numpy as np
    import math
    from sklearn.model_selection import KFold
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
    dataset=SynDataset(root='data/Rand_numnodes'+str(n_num),name='numnodes'+str(n_num))
    indx=0
    sel_feats=10
    len_mul=int(n_num/10)
    best_acc=0
    index=[idx for idx  in range(len(dataset[0].y))]
    random.shuffle(index)
    index=np.array(index)
    kf = KFold(n_splits=10)
    kf.get_n_splits(index)
    d_graph=np.zeros(10)
    acc_list=np.zeros(len(dataset))
    best_model_acc=0
    for i in range(len(dataset)):
        time=0
        data=dataset[i]
        data.x=data.x[:,:sel_feats]
        for train_index, test_index in kf.split(index):
            mask       = np.zeros(len(index))
            mask[index[train_index]]=1
            train_mask = torch.tensor(mask,dtype=torch.uint8)
            mask       = np.zeros(len(index))
            mask[index[test_index]]=1
            test_mask  = torch.tensor(mask,dtype=torch.uint8)
            val_mask   = test_mask
            model,data = locals()[Conv_Method].call(data,dataset.name,data.x.size(1),dataset.num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            best_val_acc = test_acc = 0.0
            for epoch in range(1, 201):
                train(train_mask)
                train_acc,val_acc,tmp_test_acc = test(train_mask,val_mask,test_mask)
                #log ='Epoch: 200, time: '+ str(time)  + ' acc: {:.4f} \n'
                #print((log.format(val_acc)))
                if val_acc>=best_val_acc:
                    test_acc=tmp_test_acc
                    best_val_acc = val_acc
                    if val_acc>best_model_acc:
                        best_model_acc=val_acc
                        torch.save(model.state_dict(),'bestmodel.pt')
                        log ='Epoch: 200, dataset id: '+ str(i)  + ' train_acc: {:.4f} val_acc: {:.4f} test_acc: {:.4f} \n'
                        print((log.format(train_acc,val_acc,tmp_test_acc)))
                        #test(train_mask,val_mask,test_mask)
                d_graph[time]=test_acc
            del data
            del model
            time+=1
            torch.cuda.empty_cache()
            data=dataset[i]
            data.x=data.x[:,:sel_feats]
        torch.cuda.empty_cache()
        log ='Epoch: 200, dataset id: '+ str(i)  + ' acc: {:.4f} \n'
        print((log.format(np.mean(d_graph))))
        acc_list[i]=np.mean(d_graph)
    f1=open('scores/kfold_'+Conv_Method+'_heatmap_' + '.json', 'w+')
    heat_map=np.reshape(acc_list,[int(math.sqrt(len(dataset))),int(math.sqrt(len(dataset)))])
    sio.savemat('scores/kfold_map'+Conv_Method+'.mat',{'heat_map':heat_map})
    json.dump(heat_map.tolist(),f1)


def main():
    print(sys.argv[1])
    print(sys.argv[2])
    plot_graph(sys.argv[1],int(sys.argv[2]))

if __name__=='__main__':
    main()

