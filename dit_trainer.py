import os
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from tqdm import tqdm
from torch import optim
from loaders import ATACDataset
from ae import AutoEncoder, tensor_reshape
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from diffusion import Diffusion
from transformer import DiT


class DitTrainer:
    def __init__(self, adata, model: DiT, ae: AutoEncoder = None,
                 label_name='cell_type', device='cuda', device_dis=None):
        super(DitTrainer, self).__init__()
        self.model = model
        self.ae = ae
        self.adata = adata
        self.label_name = label_name
        self.device = device
        self.device_ids = device_dis
        self.optimizer = None
        self.model = self.model.to(self.device)
        if self.ae is not None:
            self.ae = self.ae.to(self.device)
            self.ae.eval()

        self.diffusion = Diffusion(input_size=self.model.input_size, device=device)

    def save_optimizer(self, filepath):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(self.optimizer.state_dict(), filepath)

    def load_optimizer(self, filepath):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=0.01)
        self.optimizer.load_state_dict(torch.load(filepath))

    def set_ae(self, ae_model: AutoEncoder):
        self.ae = ae_model
        self.ae = self.ae.to(self.device)
        self.ae.eval()

    def load_ae(self, filepath):
        if self.ae is not None:
            checkpoint = torch.load(filepath)
            self.ae.load_state_dict(checkpoint['state_dict'])
            self.ae.ortho_tensor = checkpoint['ortho_tensor']
            self.ae.label_dict = checkpoint['label_dict']

    def save_dit(self, filepath):
        if isinstance(self.model, nn.DataParallel):
            checkpoint = {
                'state_dict': self.model.module.state_dict(),
                'label_dict': self.model.label_dict
            }
        else:
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'label_dict': self.model.label_dict
            }

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(checkpoint, filepath)

    def load_dit(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.label_dict = checkpoint['label_dict']

    def get_dit(self):
        return self.model

    def get_optim(self):
        return self.optimizer

    def train(self, dit_name='DiT', epochs=2400, save_freq=100,
              batch_size=32, lr=5e-4, weight_decay=0.01,
              save_directory=None, optim_path=None,
              zn=0., pad=0):
        device = self.device
        training_set = ATACDataset(self.adata, self.label_name, 'train', zn, pad)
        if self.model.label_dict is None:
            self.model.label_dict = training_set.label_dict
        else:
            training_set.label_dict = self.model.label_dict
        dataloader = DataLoader(training_set, batch_size=batch_size, num_workers=2, shuffle=True)

        if self.optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.device_ids is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        if self.ae is not None and self.device_ids is not None:
            self.ae = torch.nn.DataParallel(self.ae, device_ids=self.device_ids)

        mse = nn.MSELoss()

        for epoch in range(epochs):
            print(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            epoch_loss_list = list()
            self.model.train()
            for step, (x, y) in enumerate(pbar):
                x = x.to(device)
                y = y.to(device)

                if self.ae is not None:
                    with torch.no_grad():
                        x = self.ae(x, y, enc=True)

                t = self.diffusion.sample_timesteps(x.shape[0]).to(device)
                x_t, noise = self.diffusion.noise_genes(x, t)
                predicted_noise = self.model(x_t, t, y)
                loss = mse(noise, predicted_noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                epoch_loss_list.append(loss.item())

            avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
            print("Epoch: ", epoch, " Avg_MSE_loss: ", avg_epoch_loss)

            if (epoch + 1) % save_freq == 0 or save_freq == 1:
                print('Saving ckpt... ')
                dit_save_path = f'{save_directory}/{dit_name}_ep{epoch + 1}.pt'
                self.save_dit(dit_save_path)
                self.save_optimizer(optim_path)

        print('Saving ckpt... ')
        dit_save_path = f'{save_directory}/{dit_name}_ep{epochs}.pt'
        self.save_dit(dit_save_path)
        self.save_optimizer(optim_path)

    def test(self, batch_size=32,
             latent_savepath=None, result_savepath=None, zn=0., pad=0, sampler='ddim', acc_rate=10):
        device = self.device
        testing_set = ATACDataset(self.adata, self.label_name, 'test', zn, pad)
        dataloader = DataLoader(testing_set, batch_size=batch_size, num_workers=2, shuffle=False)

        if (self.device_ids is not None) and (not isinstance(self.model, nn.DataParallel)):
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        if self.ae is not None:
            raw_size = self.ae.input_dim - pad
            if (self.device_ids is not None) and (not isinstance(self.ae, nn.DataParallel)):
                self.ae = torch.nn.DataParallel(self.ae, device_ids=self.device_ids)
        else:
            raw_size = self.model.input_size - pad

        result = None
        latent_result = None

        for step, (_, y) in enumerate(dataloader):
            y = y.to(device)
            batch = len(y)
            if sampler == 'ddim':
                sub_time_seq = [i for i in range(0, 1001, acc_rate)]
                sub_time_seq.insert(1, 1)
                s = self.diffusion.sample_ddim(self.model, batch, y, sub_time_seq=sub_time_seq)
            else:
                s = self.diffusion.sample(self.model, batch, y)

            if self.ae is not None:
                with torch.no_grad():
                    if latent_savepath is not None:
                        if latent_result is None:
                            latent_result = s.to('cpu')
                        else:
                            latent_result = torch.cat((latent_result, s.to('cpu')), dim=0)

                    s = self.ae(s, y, dec=True)

            s = s[:, :, :raw_size]
            s = torch.clamp(s, min=0.)
            s = s.to('cpu')

            if result is None:
                result = s
            else:
                result = torch.cat((result, s), dim=0)
            print(result.shape)

        result = result.numpy().reshape(-1, result.shape[-1])
        # result = csr_matrix(result)
        save_adata = ad.AnnData(result)

        save_adata.obs[self.label_name] = pd.Categorical(testing_set.get_label_list())
        var_names, var_lists = testing_set.get_adata_var()
        for i in range(len(var_names)):
            save_adata.var[var_names[i]] = var_lists[i]

        print(save_adata)
        # print(save_adata.var)
        save_adata.write_h5ad(filename=result_savepath)

        if latent_result is not None:
            print(latent_result.shape)
            np.save(latent_savepath, latent_result.numpy().reshape(-1, latent_result.shape[-1]))

    def sampling(self, sample_dict, batch_size=32,
                 latent_savepath=None, result_savepath=None, zn=0., pad=0, sampler='ddim', acc_rate=10):
        device = self.device
        # make template
        total_cells = sum(sample_dict.values())
        X = np.zeros((total_cells, self.model.input_size))
        obs_data = []
        for cell_type, count in sample_dict.items():
            obs_data.extend([cell_type] * count)
        obs = pd.DataFrame({self.label_name: obs_data})
        template = ad.AnnData(X=X, obs=obs)

        sampling_set = ATACDataset(template, self.label_name, 'test', zn, pad)
        sampling_set.label_dict = self.model.label_dict
        dataloader = DataLoader(sampling_set, batch_size=batch_size, num_workers=2, shuffle=False)

        if (self.device_ids is not None) and (not isinstance(self.model, nn.DataParallel)):
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        if self.ae is not None:
            raw_size = self.ae.input_dim - pad
            if (self.device_ids is not None) and (not isinstance(self.ae, nn.DataParallel)):
                self.ae = torch.nn.DataParallel(self.ae, device_ids=self.device_ids)
        else:
            raw_size = self.model.input_size - pad

        result = None
        latent_result = None

        for step, (_, y) in enumerate(dataloader):
            y = y.to(device)
            batch = len(y)
            if sampler == 'ddim':
                sub_time_seq = [i for i in range(0, 1001, acc_rate)]
                sub_time_seq.insert(1, 1)
                s = self.diffusion.sample_ddim(self.model, batch, y, sub_time_seq=sub_time_seq)
            else:
                s = self.diffusion.sample(self.model, batch, y)

            if self.ae is not None:
                with torch.no_grad():
                    if latent_savepath is not None:
                        if latent_result is None:
                            latent_result = s.to('cpu')
                        else:
                            latent_result = torch.cat((latent_result, s.to('cpu')), dim=0)

                    s = self.ae(s, y, dec=True)

            s = s[:, :, :raw_size]
            s = torch.clamp(s, min=0.)
            s = s.to('cpu')

            if result is None:
                result = s
            else:
                result = torch.cat((result, s), dim=0)
            print(result.shape)

        result = result.numpy().reshape(-1, result.shape[-1])
        # result = csr_matrix(result)
        save_adata = ad.AnnData(result)

        save_adata.obs[self.label_name] = pd.Categorical(sampling_set.get_label_list())
        var_names, var_lists = sampling_set.get_adata_var()
        for i in range(len(var_names)):
            save_adata.var[var_names[i]] = var_lists[i]

        print(save_adata)
        # print(save_adata.var)
        save_adata.write_h5ad(filename=result_savepath)

        if latent_result is not None:
            print(latent_result.shape)
            np.save(latent_savepath, latent_result.numpy().reshape(-1, latent_result.shape[-1]))


if __name__ == '__main__':
    # Test code
    ae = AutoEncoder(
        input_dim=2000,
        latent_dim=256,
        hidden_dim=[1024 for _ in range(2)],
        embed_dim=128,
        reshape=True,
        cfg=True,
        para_devices=[0, 0]
    )
    dit = DiT(
        input_size=256,
        patch_size=2,
        hidden_size=768,
        depth=4,
        num_heads=8
    )
    adata = sc.read('dataset/data4demonstration.h5ad')

    dit_trainer = DitTrainer(adata, dit, ae)
    dit_trainer.load_ae('ae_models/demonstrate_ae_model.pt')

    dit_trainer.load_dit('dit_models/demonstrate_ep2.pt')
    dit_trainer.load_optimizer('optim/demonstrate_dit_optim.pt')

    dit_trainer.train('demonstrate', 2, save_directory='dit_models', optim_path='optim/demonstrate_dit_optim.pt')
    dit_trainer.test(latent_savepath='results/demonstrate_dit_test_latent.npy',
                     result_savepath='results/demonstrate_dit_test_result.h5ad', acc_rate=10)

    sample_dict = {'malignant': 200, 'fibroblast': 100}
    dit_trainer.sampling(sample_dict, latent_savepath='results/demonstrate_dit_sampling_latent.npy',
                         result_savepath='results/demonstrate_dit_sampling_result.h5ad', acc_rate=10)
