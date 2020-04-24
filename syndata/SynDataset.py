import torch
import os.path as osp
from torch_geometric.data import Data,InMemoryDataset,makedirs

class SynDataset(InMemoryDataset):
    def __init__(self, root,name,data_list=None, transform=None, pre_transform=None):
        self.name = name
        self.data_list=data_list
        self.files=self.processed_file_names()
        super(SynDataset, self).__init__(root)
        self.data,self.slices= torch.load(osp.join(self.processed_dir,self.files))
    def processed_file_names(self):
        return 'data.pt'
    def _download(self):
        pass
    def _process(self):
        if self.data_list is None:
            pass
        else:
            if osp.exists(osp.join(self.processed_dir,self.files)):
                return
            makedirs.makedirs(self.processed_dir)
            self.process()
    def process(self):
        # Read data into huge `Data` list.
        data, slices = self.collate(self.data_list)
        print(slices)
        torch.save((data, slices),osp.join(self.processed_dir,self.files))
#create random graph dataset
