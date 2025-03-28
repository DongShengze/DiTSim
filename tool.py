import numpy as np
import scanpy as sc
import scipy
import pandas as pd
import episcanpy as epi
import anndata as ad
import os
import scipy.stats as stats
import torch
from sklearn import metrics


def tfidf(count_mat):
    tf_mat = 1.0 * count_mat / np.tile(np.sum(count_mat, axis=0), (count_mat.shape[0], 1))
    signac_mat = np.log(1 + np.multiply(1e4 * tf_mat,
                                        np.tile((1.0 * count_mat.shape[1] / np.sum(count_mat, axis=1)).reshape(-1, 1),
                                                (1, count_mat.shape[1]))))
    return scipy.sparse.csr_matrix(signac_mat)


def compared_visualize(sample_path, target_path, label_name: str = 'cell_type', umap_name=None, tsne_name=None):
    """
    compare generated data and real data, then perform dimensionality reduction and visualization.

    Parameters:
    - sample_path: Path to the generated data (.h5ad file)
    - target_path: Path to the real data (.h5ad file)
    - label_name: Obs name of cell types. (default: 'cell_type')
    - umap_name: Path to save the UMAP plot (default: None)
    - tsne_name: Path to save the t-SNE plot (default: None)
    """
    # Load data
    sample_adata = sc.read(sample_path)
    target_adata = sc.read(target_path)

    # Extract cell type information
    sample_ct = list(sample_adata.obs[label_name].values)
    target_ct = list(target_adata.obs[label_name].values)

    # Combine cell type information
    combined_ct = target_ct + sample_ct

    # Combine matrices
    target_mat = target_adata.X.A
    sample_mat = sample_adata.X
    combined_mat = np.concatenate((target_mat, sample_mat), axis=0)

    # Create data type information
    data_type = ['target'] * target_mat.shape[0]
    data_type.extend(['sample'] * sample_mat.shape[0])

    # Create combined AnnData object
    combined_adata = ad.AnnData(combined_mat)
    combined_adata.obs[label_name] = pd.Categorical(combined_ct)
    combined_adata.obs['data_type'] = pd.Categorical(data_type)

    # Preprocess data
    epi.pp.lazy(combined_adata)

    # Perform dimensionality reduction and visualization
    epi.pl.pca_overview(combined_adata, use_raw=False)

    # Plot UMAP
    epi.pl.umap(combined_adata, color=[label_name, 'data_type'], wspace=0.4, save=f'_{umap_name}.png')

    # Plot t-SNE
    epi.pl.tsne(combined_adata, color=[label_name, 'data_type'], wspace=0.4, save=f'_{tsne_name}.png')

    # Return the combined AnnData object for further analysis
    return combined_adata


def calculate_metrics(sample_path, target_path, label_name: str = 'cell_type', calculate_kl=True):
    """
    Calculate various metrics between generated data and real data.

    Parameters:
    - sample_path: Path to the generated data (h5ad file)
    - target_path: Path to the real data (h5ad file)
    - label_name: Obs name of cell types. (default: 'cell_type')
    - calculate_kl: Whether to calculate KL divergence (requires PyTorch) (default: True)

    Returns:
    - A dictionary containing the calculated metrics:
        - 'SCC': Spearman's Correlation Coefficient between overall means
        - 'PCC': Pearson's Correlation Coefficient between overall means
        - 'KL_Div': Kullback-Leibler Divergence (if calculate_kl is True)
        - 'SCC_per_cell_type': Average SCC per cell type
        - 'PCC_per_cell_type': Average PCC per cell type
    """
    # Load data
    sample = sc.read(sample_path)
    target = sc.read(target_path)

    # Extract matrices
    x = sample.X
    y = target.X.A

    # Calculate overall means
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)

    # Calculate SCC and PCC for overall means
    scc = stats.spearmanr(x_mean, y_mean).correlation
    pcc = stats.pearsonr(x_mean, y_mean)[0]

    metrics = {
        'SCC': scc,
        'PCC': pcc
    }

    # Calculate KL Divergence if required
    if calculate_kl:
        sample_mat = torch.Tensor(x)
        target_mat = torch.Tensor(y)
        sample_mat = torch.nn.functional.log_softmax(sample_mat, dim=1)
        target_mat = torch.nn.functional.softmax(target_mat, dim=1)
        kl_div = torch.nn.functional.kl_div(sample_mat, target_mat, reduction='batchmean')
        metrics['KL_Div'] = kl_div.item()

    # Calculate metrics per cell type
    unique_cell_type = sample.obs[label_name].unique()

    scc_list = []
    pcc_list = []

    for cell_type in unique_cell_type:
        sample_index = sample.obs[label_name] == cell_type
        target_index = target.obs[label_name] == cell_type

        # Check if cell type exists in both datasets
        if not np.any(sample_index) or not np.any(target_index):
            continue

        sample_cell = np.mean(x[sample_index], axis=0)
        target_cell = np.mean(y[target_index], axis=0)

        scc_list.append(stats.spearmanr(sample_cell, target_cell).correlation)
        pcc_list.append(stats.pearsonr(sample_cell, target_cell)[0])

    # Calculate averages
    if scc_list:
        scc_ct = sum(scc_list) / len(scc_list)
        pcc_ct = sum(pcc_list) / len(pcc_list)
        metrics['SCC_per_cell_type'] = scc_ct
        metrics['PCC_per_cell_type'] = pcc_ct
    else:
        metrics['SCC_per_cell_type'] = None
        metrics['PCC_per_cell_type'] = None

    return metrics


def calculate_clustering_metrics(sample_path, target_path, label_name: str = 'cell_type', cluster_method='leiden',
                                 n_clusters=None, save_plots=False,
                                 ):
    """
    Calculate clustering metrics (ARI and AMI) for generated and real data using specified clustering method.

    Parameters:
    - sample_path: Path to the generated data (h5ad file)
    - target_path: Path to the real data (h5ad file)
    - label_name: Obs name of cell types. (default: 'cell_type')
    - cluster_method: Clustering method to use ('louvain' or 'leiden') (default: 'leiden')
    - n_clusters: Number of clusters to use for clustering (default: None, will be determined from cell types)
    - save_plots: Whether to save UMAP plots of clustering results (default: False)
    - plot_dir: Directory to save plots (required if save_plots is True)

    Returns:
    - A dictionary containing:
        - 'sample_ari': Adjusted Rand Index for generated data
        - 'sample_ami': Adjusted Mutual Information for generated data
        - 'target_ari': Adjusted Rand Index for real data
        - 'target_ami': Adjusted Mutual Information for real data
    """
    # Load data
    sample = sc.read(sample_path)
    target = sc.read(target_path)

    # Determine number of clusters if not provided
    if n_clusters is None:
        n_clusters = len(pd.unique(sample.obs[label_name].values).tolist())

    # Process and cluster sample data
    sample_ari, sample_ami = clustering(sample, label_name, cluster_method, n_clusters, save_plots,
                                        'sample')

    # Process and cluster target data
    target_ari, target_ami = clustering(target, label_name, cluster_method, n_clusters, save_plots,
                                        'target')

    return {
        'sample_ari': sample_ari,
        'sample_ami': sample_ami,
        'target_ari': target_ari,
        'target_ami': target_ami
    }


def clustering(adata, label_name, cluster_method, n_clusters, save_plots, data_type):
    """
    Helper function to process, cluster, and calculate metrics for a single dataset.
    """
    # Preprocess data
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.tsne(adata)
    sc.tl.umap(adata)

    # Perform clustering
    if cluster_method == 'louvain':
        sc.tl.louvain(adata)
        cluster_key = 'louvain'
    elif cluster_method == 'leiden':
        sc.tl.leiden(adata)
        cluster_key = 'leiden'
    else:
        raise ValueError("Unsupported clustering method. Use 'louvain' or 'leiden'.")

    # Optionally enforce number of clusters using episcanpy
    epi.tl.getNClusters(adata, n_cluster=n_clusters, method=cluster_method, key_added=f"{cluster_method}_{n_clusters}")
    cluster_key = f"{cluster_method}_{n_clusters}"

    # Prepare cell type labels
    cell_types = adata.obs[label_name].values
    unique_ct = pd.unique(cell_types).tolist()
    ct_labels = [unique_ct.index(ct) for ct in cell_types]

    # Get cluster labels
    cluster_labels = adata.obs[cluster_key].values

    # Calculate metrics
    ari = metrics.adjusted_rand_score(ct_labels, cluster_labels)
    ami = metrics.adjusted_mutual_info_score(ct_labels, cluster_labels)

    # Save UMAP plot if required
    if save_plots:
        sc.pl.umap(adata, color=[cluster_key, label_name], save=f'_{data_type}_{cluster_method}.png')

    return ari, ami


if __name__ == '__main__':
    # metrics = calculate_metrics('results/demonstrate_dit_test_result.h5ad', 'dataset/data4demonstration.h5ad')
    # print(metrics)
    metrics = calculate_clustering_metrics('results/demonstrate_dit_test_result.h5ad',
                                           'dataset/data4demonstration.h5ad',
                                           save_plots=True, cluster_method='louvain')
    print(metrics)
