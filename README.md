# DiTSim: A Diffusion Transformers-Based single-cell ATAC-seq Data Simulator

## Preparation

### Configurate environment

```conda env create -f ditsim.yml```

### Data preprocess

We need a `.h5ad` file of scATAC-seq dataset with `obs` of cell-type.

```python
import scanpy as sc

# for convenience, we use a small toy dataset here.
adata = sc.read('dataset/data4demonstration.h5ad')
```

Before start training, use `tfidf()` to preprocess `adata` if your count mat is binary data.

```python
from tool import tfidf

adata.X = tfidf(adata.X.T).T
```

More functions for data analysis after simulating please find `data_analysis.py`.

## Training Models
Find `quick_start.py` to get an overview of the utilization of DiTSim.

### Autoencoder Training

#### Initialize AutoEncoder and Trainer
```python
from ae_trainer import AutoencoderTrainer
from ae import AutoEncoder

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
ae_trainer = AutoencoderTrainer(
        model=ae, 
        adata=adata,
        label_name='cell_type',  # obs name for cell-type label in .h5ad file
        device='cuda'
    )
```
#### Train the Model
```python
# start training
ae_trainer.train(
        epochs=200,  # training epochs
        end_loss=0.01,  # stop training when loss lower than this
        batch_size=32,
        lr=5e-4,
        weight_decay=0.01,
        # cs_loss controls the latent space distribution, higher for greater discrimination between cell-types.
        cs_loss_ratio=0.,  # [0, 1] float. Set 0. if you don't need higher ARI/AMI metrics.
        save_freq=10,  # save checkpoint frequency
        save_path='ae_models/demonstrate_ae_model.pt', 
        optim_path='optim/demonstrate_ae_optim.pt'
    )
```
#### Test the Model
```python
# test ae after training
ae_trainer.test(
        latent_savepath='results/demonstrate_ae_latent.npy', 
        recon_savepath='results/demonstrate_ae_recon.npy'
    )
```

### DiT Training
#### Initialize DiT and Trainer
```python
from transformer import DiT
from dit_trainer import DitTrainer

# define dit model
dit = DiT(
        input_size=256,  # should be same as ae latent_dim
        patch_size=2,  # smaller for more tokens in transformer (higher cost & better performance)
        hidden_size=768,  # token length
        depth=4,  # num of dit blocks
        num_heads=8  # sa head num
    )

dit_trainer = DitTrainer(
        adata=adata,  # dataset
        model=dit,  # DiT model definition
        ae=ae,  # AE model definition
        label_name='cell_type',
        device='cuda:0',
        device_dis=None  # input a list if you have more than 1 GPU for training
    )  
```
#### Load Pretrained AE Model
```python
# load well-fitted ae model
dit_trainer.load_ae('ae_models/demonstrate_ae_model.pt')
```
#### Train the Model
```python
# start training
dit_trainer.train(
        dit_name='demonstrate',
        epochs=1600,
        save_freq=100,
        batch_size=32,
        lr=1e-4,
        weight_decay=0.01,
        save_directory='dit_models',  # directory for saving dit models.
        optim_path='optim/demonstrate_dit_optim.pt'
    )
```
#### Test the Model
```python
# test dit
# Generate simulated data of the same size as the dataset.
dit_trainer.test(
        batch_size=32,
        latent_savepath='results/demonstrate_dit_test_latent.npy',
        result_savepath='results/demonstrate_dit_test_result.h5ad', 
        sampler='ddim',  # choose ddim or ddpm for sampling
        acc_rate=10
    )
```

## Simulating
```python
# make a dict of {'cell type': num, ...}
sample_dict = {'malignant': 200, 'fibroblast': 100}
# in this example, we generate 200*malignant and 100*fibroblast simulated data.
dit_trainer.sampling(
        sample_dict, 
        latent_savepath='results/demonstrate_dit_sampling_latent.npy',
        result_savepath='results/demonstrate_dit_sampling_result.h5ad', 
        sampler='ddim',  # choose ddim or ddpm for sampling
        acc_rate=10
    )
```

## User Interface Details

### Model Definition

#### Autoencoder
The `AutoEncoder` class is defined in the `ae.py` file. Below are the parameters for the `AutoEncoder` model:

- **input_dim**: The dimensionality of the input data.
- **hidden_dim**: A list specifying the number of neurons in each hidden layer (default: `[1024, 1024]`).
- **latent_dim**: The dimensionality of the latent space (default: `128`).
- **input_dropout**: Dropout rate for the input layer (default: `0.4`).
- **dropout**: Dropout rate for the hidden layers (default: `0.5`).
- **residual**: Whether to use residual connections (default: `True`).
- **reshape**: Whether to reshape the input data (default: `True`).
- **num_classes**: The number of classes for label embedding (default: `100`).
- **embed_dim**: The dimensionality of the label embedding (default: `128`).
- **class_dropout_prob**: Dropout probability for class labels (default: `0.`).
- **cfg**: Whether to use classifier-free guidance (default: `True`).
- **para_devices**: A list of devices for parallel training (default: `None`).
```python
# Example Usage
ae = AutoEncoder(
    input_dim=2000,  # Dimensionality of the input data
    latent_dim=256,  # Latent space dimensionality
    hidden_dim=[1024, 1024],  # Hidden layers configuration
    embed_dim=128,  # Label embedding dimensionality
    reshape=True,  # Reshape the input data
    cfg=True,  # Use classifier-free guidance
    para_devices=[0, 0]  # Use two GPUs for parallel training
)
```

#### Diffusion Transformer (DiT)
The `DiT` class is defined in the `transformer.py` file. Below are the parameters for the `DiT` model:

- **input_size**: The input size of the data (default: `2000`).
- **patch_size**: The size of each patch (default: `10`).
- **in_channels**: The number of input channels (default: `1`).
- **hidden_size**: The hidden size of the transformer (default: `1024`).
- **depth**: The number of transformer blocks (default: `1`).
- **num_heads**: The number of attention heads (default: `16`).
- **mlp_ratio**: The ratio of the MLP layer (default: `4.0`).
- **class_dropout_prob**: Dropout probability for class labels (default: `0.1`).
- **num_classes**: The number of classes for label embedding (default: `100`).

```python
# Example Usage
dit = DiT(
    input_size=256,  # Should match the latent dimension of the AutoEncoder
    patch_size=2,  # Smaller for more tokens in the transformer
    hidden_size=768,  # Token length
    depth=4,  # Number of DiT blocks
    num_heads=8  # Number of self-attention heads
)
```

### AutoencoderTrainer (`ae_trainer.py`)

#### Methods

- `__init__(self, model: AutoEncoder, adata, label_name='cell_type', device='cuda', para=None)`

  - **Description**: Initializes the trainer with the model, dataset, label name, device, and parallel devices.
  - **Parameters**:
    - `model`: The AutoEncoder model.
    - `adata`: The dataset in AnnData format.
    - `label_name`: The name of the cell-type label in the dataset's `obs` (default: `'cell_type'`).
    - `device`: The device to use for training (default: `'cuda'`).
    - `para`: Parallel devices configuration (default: `None`).

- `save_optimizer(self, filepath)`

  - **Description**: Saves the optimizer state dictionary to the specified file path.
  - **Parameters**:
    - `filepath`: The file path to save the optimizer state.

- `load_optimizer(self, filepath)`

  - **Description**: Loads the optimizer state dictionary from the specified file path.
  - **Parameters**:
    - `filepath`: The file path to load the optimizer state.

- `save_ae(self, filepath)`

  - **Description**: Saves the AutoEncoder model state dictionary, ortho_tensor, and label_dict to the specified file path.
  - **Parameters**:
    - `filepath`: The file path to save the model.

- `load_ae(self, filepath)`

  - **Description**: Loads the AutoEncoder model state dictionary, ortho_tensor, and label_dict from the specified file path.
  - **Parameters**:
    - `filepath`: The file path to load the model.

- `get_model(self)`

  - **Description**: Returns the AutoEncoder model.
  - **Return**: The AutoEncoder model.

- `get_optim(self)`

  - **Description**: Returns the optimizer.
  - **Return**: The optimizer.

- `train(self, epochs=200, end_loss=0.01, batch_size=32, lr=5e-4, weight_decay=0.01, cs_loss_ratio=0., save_freq=10, save_path=None, optim_path=None, zn=0., pad=0)`

  - **Description**: Trains the AutoEncoder model with the specified parameters.
  - **Parameters**:
    - `epochs`: Number of training epochs (default: `200`).
    - `end_loss`: Stop training when loss is lower than this value (default: `0.01`).
    - `batch_size`: Batch size for training (default: `32`).
    - `lr`: Learning rate (default: `5e-4`).
    - `weight_decay`: Weight decay (default: `0.01`).
    - `cs_loss_ratio`: Cosine similarity loss ratio (default: `0.`).
    - `save_freq`: Frequency to save checkpoints (default: `10`).
    - `save_path`: File path to save the model (default: `None`).
    - `optim_path`: File path to save the optimizer (default: `None`).
    - `zn`: Zero noise parameter (default: `0.`).
    - `pad`: Padding parameter (default: `0`).

- `test(self, batch_size=32, latent_savepath=None, recon_savepath=None, zn=0., pad=0)`

  - **Description**: Tests the AutoEncoder model and saves the latent representations and reconstructions if specified.
  - **Parameters**:
    - `batch_size`: Batch size for testing (default: `32`).
    - `latent_savepath`: File path to save latent representations (default: `None`).
    - `recon_savepath`: File path to save reconstructions (default: `None`).
    - `zn`: Zero noise parameter (default: `0.`).
    - `pad`: Padding parameter (default: `0`).

### DitTrainer (`dit_trainer.py`)

#### Methods

- `__init__(self, adata, model: DiT, ae: AutoEncoder = None, label_name='cell_type', device='cuda', device_dis=None)`

  - **Description**: Initializes the trainer with the dataset, DiT model, AE model, label name, device, and parallel devices.
  - **Parameters**:
    - `adata`: The dataset in AnnData format.
    - `model`: The DiT model.
    - `ae`: The AutoEncoder model (default: `None`).
    - `label_name`: The name of the cell-type label in the dataset's `obs` (default: `'cell_type'`).
    - `device`: The device to use for training (default: `'cuda'`).
    - `device_dis`: Parallel devices configuration (default: `None`).

- `save_optimizer(self, filepath)`

  - **Description**: Saves the optimizer state dictionary to the specified file path.
  - **Parameters**:
    - `filepath`: The file path to save the optimizer state.

- `load_optimizer(self, filepath)`

  - **Description**: Loads the optimizer state dictionary from the specified file path.
  - **Parameters**:
    - `filepath`: The file path to load the optimizer state.

- `set_ae(self, ae_model: AutoEncoder)`

  - **Description**: Sets the AE model for the trainer.
  - **Parameters**:
    - `ae_model`: The AutoEncoder model.

- `load_ae(self, filepath)`

  - **Description**: Loads the AE model state dictionary from the specified file path.
  - **Parameters**:
    - `filepath`: The file path to load the AE model.

- `save_dit(self, filepath)`

  - **Description**: Saves the DiT model state dictionary and label_dict to the specified file path.
  - **Parameters**:
    - `filepath`: The file path to save the DiT model.

- `load_dit(self, filepath)`

  - **Description**: Loads the DiT model state dictionary and label_dict from the specified file path.
  - **Parameters**:
    - `filepath`: The file path to load the DiT model.

- `get_dit(self)`

  - **Description**: Returns the DiT model.
  - **Return**: The DiT model.

- `get_optim(self)`

  - **Description**: Returns the optimizer.
  - **Return**: The optimizer.

- `train(self, dit_name='DiT', epochs=2400, save_freq=100, batch_size=32, lr=5e-4, weight_decay=0.01, save_directory=None, optim_path=None, zn=0., pad=0)`

  - **Description**: Trains the DiT model with the specified parameters.
  - **Parameters**:
    - `dit_name`: Name of the DiT model (default: `'DiT'`).
    - `epochs`: Number of training epochs (default: `2400`).
    - `save_freq`: Frequency to save checkpoints (default: `100`).
    - `batch_size`: Batch size for training (default: `32`).
    - `lr`: Learning rate (default: `5e-4`).
    - `weight_decay`: Weight decay (default: `0.01`).
    - `save_directory`: Directory to save the model (default: `None`).
    - `optim_path`: File path to save the optimizer (default: `None`).
    - `zn`: Zero noise parameter (default: `0.`).
    - `pad`: Padding parameter (default: `0`).

- `test(self, batch_size=32, latent_savepath=None, result_savepath=None, zn=0., pad=0, sampler='ddim', acc_rate=10)`

  - **Description**: Tests the DiT model and saves the results if specified.
  - **Parameters**:
    - `batch_size`: Batch size for testing (default: `32`).
    - `latent_savepath`: File path to save latent representations (default: `None`).
    - `result_savepath`: File path to save results (default: `None`).
    - `zn`: Zero noise parameter (default: `0.`).
    - `pad`: Padding parameter (default: `0`).
    - `sampler`: Sampling method (`'ddim'` or `'ddpm'`) (default: `'ddim'`).
    - `acc_rate`: Acceleration rate for sampling (default: `10`).

- `sampling(self, sample_dict, batch_size=32, latent_savepath=None, result_savepath=None, zn=0., pad=0, sampler='ddim', acc_rate=10)`

  - **Description**: Generates simulated data based on the specified sample dictionary and saves the results if specified.
  - **Parameters**:
    - `sample_dict`: Dictionary specifying the number of cells to generate for each cell type.
    - `batch_size`: Batch size for sampling (default: `32`).
    - `latent_savepath`: File path to save latent representations (default: `None`).
    - `result_savepath`: File path to save results (default: `None`).
    - `zn`: Zero noise parameter (default: `0.`).
    - `pad`: Padding parameter (default: `0`).
    - `sampler`: Sampling method (`'ddim'` or `'ddpm'`) (default: `'ddim'`).
    - `acc_rate`: Acceleration rate for sampling (default: `10`).

## Tool Functions (`tool.py`)

### Data Visualization

#### `compared_visualize`
```python
compared_visualize(sample_path, target_path, label_name='cell_type', umap_name=None, tsne_name=None)
```

- **Description**: Compare generated data and real data, then perform dimensionality reduction and visualization.
- **Parameters**:
  - `sample_path`: Path to the generated data (.h5ad file).
  - `target_path`: Path to the real data (.h5ad file).
  - `label_name`: Obs name of cell types (default: `'cell_type'`).
  - `umap_name`: Fig name for saving the UMAP plot (default: `None`).
  - `tsne_name`: Fig name for saving the t-SNE plot (default: `None`).
- **Returns**: Combined AnnData object for further analysis.

### Metrics Calculation

#### `calculate_metrics`
```python
calculate_metrics(sample_path, target_path, label_name: str = 'cell_type', calculate_kl=True)
```

- **Description**: Calculate various metrics between generated data and real data.
- **Parameters**:
  - `sample_path`: Path to the generated data (.h5ad file).
  - `target_path`: Path to the real data (.h5ad file).
  - `label_name`: Obs name of cell types (default: `'cell_type'`).
  - `calculate_kl`: Whether to calculate KL divergence (requires PyTorch) (default: `True`).
- **Returns**: A dictionary containing the calculated metrics:
  - `'SCC'`: Spearman's Correlation Coefficient between overall means.
  - `'PCC'`: Pearson's Correlation Coefficient between overall means.
  - `'KL_Div'`: Kullback-Leibler Divergence (if `calculate_kl` is `True`).
  - `'SCC_per_cell_type'`: Average SCC per cell type.
  - `'PCC_per_cell_type'`: Average PCC per cell type.

### Clustering Metrics Calculation

#### `calculate_clustering_metrics`
```python
calculate_clustering_metrics(sample_path, target_path, 
                             label_name='cell_type', cluster_method='leiden', 
                             n_clusters=None, save_plots=False)
```

- **Description**: Calculate clustering metrics (ARI and AMI) for generated and real data using the specified clustering method.
- **Parameters**:
  - `sample_path`: Path to the generated data (h5ad file).
  - `target_path`: Path to the real data (h5ad file).
  - `label_name`: Obs name of cell types (default: `'cell_type'`).
  - `cluster_method`: Clustering method to use (`'louvain'` or `'leiden'`) (default: `'leiden'`).
  - `n_clusters`: Number of clusters to use for clustering (default: `None`, will be determined from cell types).
  - `save_plots`: Whether to save UMAP plots of clustering results (default: `False`).
- **Returns**: A dictionary containing:
  - `'sample_ari'`: Adjusted Rand Index for generated data.
  - `'sample_ami'`: Adjusted Mutual Information for generated data.
  - `'target_ari'`: Adjusted Rand Index for real data.
  - `'target_ami'`: Adjusted Mutual Information for real data.