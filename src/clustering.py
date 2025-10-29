import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import umap
import hdbscan


class Clustering:
    def __init__(self):
        self.videos_embedded = pd.read_csv("../data/videos_embedded.csv")
        self.embeddings = np.vstack(self.videos_embedded["embedding"].apply(ast.literal_eval).values)
    
    def dim_reduction(self):
        self.umap_model = umap.UMAP(n_neighbors = 200, n_components = 2, metric = "cosine")
        self.embeddings_umap = self.umap_model.fit_transform(self.embeddings)
    
    def umap_clustering(self):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size = 15, metric = "euclidean")
        self.labels = self.clusterer.fit_predict(self.embeddings_umap)

    def plot_clustering(self):
        plt.figure(figsize=(10,7))
        plt.scatter(self.embeddings_umap[:,0], self.embeddings_umap[:,1], c=self.labels, s=10)
        plt.title("Clusters of Text Chunks (UMAP + HDBSCAN)")
        plt.show()
