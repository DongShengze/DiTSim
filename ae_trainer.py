import os
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
from tqdm import tqdm
from torch import optim
from loaders import ATACDataset
from ae import AutoEncoder, CosineSimilarityLoss, tensor_reshape
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random


class AutoencoderTrainer:
    def __init__(self, model: AutoEncoder, adata, label_name='cell_type', device='cuda', para=None):
        super(AutoencoderTrainer, self).__init__()
        self.model = model
        self.adata = adata
        self.label_name = label_name
        self.device = device
        self.para = para
        self.optimizer = None
        self.model = self.model.to(self.device)
        if self.para is not None:
            self.model.para(self.para)

    def save_optimizer(self, filepath):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(self.optimizer.state_dict(), filepath)

    def load_optimizer(self, filepath):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=0.01)
        self.optimizer.load_state_dict(torch.load(filepath))

    def save_ae(self, filepath):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'ortho_tensor': self.model.ortho_tensor,
            'label_dict': self.model.label_dict
        }

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(checkpoint, filepath)

    def load_ae(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.ortho_tensor = checkpoint['ortho_tensor']
        self.model.label_dict = checkpoint['label_dict']

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optimizer

    def train(self, epochs=200, end_loss=0.01,
              batch_size=32, lr=5e-4, weight_decay=0.01, cs_loss_ratio=0.,
              save_freq=10, save_path=None, optim_path=None,
              zn=0., pad=0):
        if self.optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        training_set = ATACDataset(self.adata, self.label_name, 'train', zn, pad)
        dataloader = DataLoader(training_set, batch_size=batch_size, num_workers=2, shuffle=True)

        if self.model.label_dict is None:
            self.model.label_dict = training_set.label_dict
        else:
            training_set.label_dict = self.model.label_dict

        device = self.device

        mse = nn.MSELoss()
        cos_sim = CosineSimilarityLoss()
        ot_target = self.model.ortho_tensor.to(device)

        for epoch in range(epochs):
            print(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            epoch_loss_list = list()
            self.model.train()

            for step, (x, y) in enumerate(pbar):
                x = x.to(device)
                y = y.to(device)
                latent, recon = self.model(x, y)

                mse_loss = mse(recon, x)
                if cs_loss_ratio > 0.:
                    cs_loss = cos_sim(ot_target[y], tensor_reshape(latent))
                    loss = mse_loss + cs_loss_ratio * cs_loss
                else:
                    loss = mse_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(LOSS=loss.item())
                epoch_loss_list.append(mse_loss.item())

            avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
            print("Epoch: ", epoch, " Avg_MSE_loss: ", avg_epoch_loss)

            if (epoch + 1) % save_freq == 0 or save_freq == 1:
                print('Saving ckpt ...')
                self.save_ae(save_path)
                self.save_optimizer(optim_path)

            if avg_epoch_loss < end_loss:
                print('Hit ending loss and stop training. Saving ckpt ...')
                self.save_ae(save_path)
                self.save_optimizer(optim_path)

        print('Saving ckpt ...')
        self.save_ae(save_path)
        self.save_optimizer(optim_path)

    def test(self, batch_size=32, latent_savepath=None, recon_savepath=None, zn=0., pad=0):
        testing_set = ATACDataset(self.adata, self.label_name, 'test', zn, pad)
        dataloader = DataLoader(testing_set, batch_size=batch_size, num_workers=2, shuffle=False)

        mse_loss = nn.MSELoss()
        save_arr = None
        latent_arr = None
        pbar = tqdm(dataloader)
        epoch_test_loss = list()
        device = self.device
        self.model.eval()

        with torch.no_grad():
            print('Testing Autoencoder...')
            for step, (x, y) in enumerate(pbar):
                x = x.to(device)
                y = y.to(device)

                latent, recon = self.model(x, y)
                loss = mse_loss(recon, x)

                pbar.set_postfix(MSE=loss.item())
                epoch_test_loss.append(loss.item())

                recon = torch.clamp(recon, min=0.0)
                recon = recon.reshape(-1, recon.shape[-1])
                recon = recon[:, :(self.model.input_dim - pad)]
                latent = latent.reshape(-1, latent.shape[-1])

                if save_arr is None:
                    save_arr = recon.to('cpu')
                    latent_arr = latent.to('cpu')
                else:
                    save_arr = np.concatenate((save_arr, recon.to('cpu')), axis=0)
                    latent_arr = np.concatenate((latent_arr, latent.to('cpu')), axis=0)

            avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
            print("Test loss: ", avg_test_loss)

        if recon_savepath is not None:
            directory = os.path.dirname(recon_savepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
            np.save(recon_savepath, save_arr)
            print(save_arr.shape)

        if latent_savepath is not None:
            directory = os.path.dirname(latent_savepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
            np.save(latent_savepath, latent_arr)
            print(latent_arr.shape)


if __name__ == '__main__':
    # Test code
    ae = AutoEncoder(
        input_dim=2000,
        latent_dim=256,
        hidden_dim=[1024 for _ in range(2)],
        embed_dim=128,
        reshape=True,
        cfg=True,
    )
    adata = sc.read('dataset/data4demonstration.h5ad')

    ae_trainer = AutoencoderTrainer(ae, adata)

    ae_trainer.load_ae('ae_models/demonstrate_ae_model.pt')
    ae_trainer.load_optimizer('optim/demonstrate_ae_optim.pt')

    ae_trainer.train(10, save_path='ae_models/demonstrate_ae_model.pt', optim_path='optim/demonstrate_ae_optim.pt')
    ae_trainer.test(latent_savepath='results/demonstrate_ae_latent.npy', recon_savepath='results/demonstrate_ae_recon.npy')


