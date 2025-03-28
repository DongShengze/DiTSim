import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader


class ATACDataset(Dataset):
    def __init__(self,
                 input_data=None,
                 label_name='cell_type',
                 flag='train',
                 zn=0.,
                 pad=0,
                 masked_code=100,
                 sample_dict=None):
        assert flag in ['train', 'test', 'eval']
        if (flag == 'train' or flag == 'test') and input_data is None:
            print('[ERROR] No anndata given!')
        elif flag == 'eval' and sample_dict is None:
            print('[ERROR] No cell type dict given!')
        self.flag = flag
        self.data = input_data
        self.zn = zn
        self.label_name = label_name
        self.pad = int(pad)
        self.masked_code = masked_code
        self.mat, self.label_list, self.label_dict = self.load_adata(self.data, self.label_name)

    def load_adata(self, adata, label_name):
        mat = adata.X  # (m, n) ndarray
        if type(mat) == np.ndarray:
            pass
        else:
            mat = mat.A
        mat = mat.astype(np.float32)

        if self.pad > 0:
            p = torch.zeros(mat.shape[0], self.pad)
            mat = torch.Tensor(mat)
            mat = torch.cat((mat, p), dim=1)

        if self.zn < 0.:
            mat[mat == 0.] = self.zn

        label_list = list(adata.obs[label_name].values)
        types = pd.unique(adata.obs[label_name].values)
        type_num = [i for i in range(len(types))]
        label_dict = dict(zip(types, type_num))
        label_dict['masked'] = self.masked_code

        return mat, label_list, label_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.mat[index][None, :]
        label = self.label_list[index]
        y = self.label_dict[label]

        return x, y

    def get_label_dict(self):
        return self.label_dict

    def set_label_dict(self, new_dict):
        self.label_dict = new_dict

    def get_label_list(self):
        return self.label_list

    def mask_label_from_dict(self, label_key):
        self.label_dict[label_key] = self.masked_code

    def get_adata_var(self):
        adata = self.data
        var = adata.var.values
        var_names = adata.var.keys().values.tolist()
        var_list = []
        for i in range(var.shape[1]):
            v = var[:, i].tolist()
            var_list.append(v)
        return var_names, var_list


if __name__ == '__main__':
    # test code
    label_name = 'cell_type'
    adata = sc.read('dataset/data4demonstration.h5ad')

    dataset = ATACDataset(adata)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=True)

    for step, (x, y) in enumerate(dataloader):
        print('step is: ', step)
        print(x.shape)
        print(x)
        print(y)




