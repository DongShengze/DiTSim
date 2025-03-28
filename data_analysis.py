import tool

if __name__ == '__main__':

    # compared visualize
    combined_data = tool.compared_visualize(
        sample_path='results/demonstrate_dit_test_result.h5ad',
        target_path='dataset/data4demonstration.h5ad',
        label_name='cell_type',
        umap_name='umap_plot',
        tsne_name='tsne_plot'
    )

    metrics = tool.calculate_metrics(
        sample_path='results/demonstrate_dit_test_result.h5ad',
        target_path='dataset/data4demonstration.h5ad',
        label_name='cell_type',
        calculate_kl=True
    )
    print("Calculated Metrics:")
    print(metrics)

    clustering_metrics = tool.calculate_clustering_metrics(
        sample_path='results/demonstrate_dit_test_result.h5ad',
        target_path='dataset/data4demonstration.h5ad',
        label_name='cell_type',
        cluster_method='leiden',
        n_clusters=None,
        save_plots=True
    )
    print("Clustering Metrics:")
    print(clustering_metrics)
