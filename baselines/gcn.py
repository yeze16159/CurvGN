import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True)
        self.conv2 = GCNConv(16, num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=0.6,training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def call(data,name,num_features,num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model= Net(data,num_features,num_classes).to(device)
    return model, data

