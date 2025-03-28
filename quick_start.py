import scanpy as sc
from ae_trainer import AutoencoderTrainer
from ae import AutoEncoder
from transformer import DiT
from dit_trainer import DitTrainer

if __name__ == '__main__':
    # for convenience, we use a small toy dataset here.
    adata = sc.read('dataset/data4demonstration.h5ad')

    ## train autoencoder
    # define ae model
    ae = AutoEncoder(
        input_dim=2000,  # should be same as the dimensionality of atac sample
        latent_dim=256,  # latent dimensionality
        hidden_dim=[1024 for _ in range(2)],  # list for hidden layers
        embed_dim=128,  # for condition embedding, ignore this if you just want unsupervised ae
        reshape=True,
        cfg=True,  # unsupervised ae model
    )
    # define ae_trainer
    ae_trainer = AutoencoderTrainer(model=ae, adata=adata,
                                    label_name='cell_type',  # obs name for cell-type label in .h5ad file
                                    device='cuda')

    # start training
    ae_trainer.train(epochs=200,  # training epochs
                     end_loss=0.01,  # stop training when loss lower than this
                     batch_size=32,
                     lr=5e-4,
                     weight_decay=0.01,
                     # cs_loss controls the latent space distribution, higher for greater discrimination between cell-types.
                     cs_loss_ratio=0.,  # [0, 1] float. Set 0. if you don't need higher ARI/AMI metrics.
                     save_freq=10,  # save checkpoint frequency
                     save_path='ae_models/demonstrate_ae_model.pt',
                     optim_path='optim/demonstrate_ae_optim.pt')
    # test ae after training
    ae_trainer.test(latent_savepath='results/demonstrate_ae_latent.npy',
                    recon_savepath='results/demonstrate_ae_recon.npy')

    ## train DiT
    # define dit model
    dit = DiT(
        input_size=256,  # should be same as ae latent_dim
        patch_size=2,  # smaller for more tokens in transformer (higher cost & better performance)
        hidden_size=768,  # token length
        depth=4,  # num of dit blocks
        num_heads=8  # sa head num
    )

    dit_trainer = DitTrainer(adata=adata,  # dataset
                             model=dit,  # DiT model definition
                             ae=ae,  # AE model definition
                             label_name='cell_type',
                             device='cuda:0',
                             device_dis=None)  # input a list if you have more than 1 GPU for training
    # load well-fitted ae model
    dit_trainer.load_ae('ae_models/demonstrate_ae_model.pt')

    # start training
    dit_trainer.train(dit_name='demonstrate',
                      epochs=2400,  # training epochs
                      save_freq=100,
                      batch_size=32,
                      lr=1e-4,
                      weight_decay=0.01,
                      save_directory='dit_models',  # directory for saving dit models.
                      optim_path='optim/demonstrate_dit_optim.pt')
    # test dit
    # Generate simulated data of the same size as the dataset.
    dit_trainer.test(batch_size=32,
                     latent_savepath='results/demonstrate_dit_test_latent.npy',
                     result_savepath='results/demonstrate_dit_test_result.h5ad',
                     sampler='ddim',  # choose ddim or ddpm for sampling
                     acc_rate=10)

    ## sampling
    # make a dict of {'cell type': num, ...}
    sample_dict = {'malignant': 200, 'fibroblast': 100}
    # in this example, we generate 200*malignant and 100*fibroblast simulated data.
    dit_trainer.sampling(sample_dict,
                         latent_savepath='results/demonstrate_dit_sampling_latent.npy',
                         result_savepath='results/demonstrate_dit_sampling_result.h5ad',
                         sampler='ddim',  # choose 'ddim' or 'ddpm' for sampling
                         acc_rate=10)
