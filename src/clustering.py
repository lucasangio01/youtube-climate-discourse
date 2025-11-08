import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
from itertools import combinations
import networkx as nx
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self):
        self.videos_classified = pd.read_csv("../data/videos_classified.csv")
        self.embeddings = self.videos_classified["embedding"].values
        self.umap_model = umap.UMAP(n_neighbors = 30, n_components = 2, metric = "cosine")
        self.embeddings_umap = self.umap_model.fit_transform(self.embeddings)

    def hdbscan_clustering(self):
        self.hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size = 8, min_samples = 5, metric = "euclidean")
        self.hdbscan_labels = self.hdbscan_clusterer.fit_predict(self.embeddings_umap)

    def plot_kmeans(self):
        kmeans = KMeans(n_clusters = 2, max_iter = 100, n_init = 1)
        kmeans_labels = kmeans.fit_predict(self.embeddings_umap)
        plt.figure(figsize = (10, 7))
        plt.scatter(self.embeddings_umap[:, 0], self.embeddings_umap[:, 1], c = kmeans_labels, s = 10, alpha = 0.8, cmap = "Paired")
        plt.title(f"KMeans Clustering (k={k}) on UMAP (2D)")
        plt.show()
        return kmeans_labels

    def plot_hdbscan(self):
        plt.figure(figsize = (10,7))
        plt.scatter(self.embeddings_umap[:,0], self.embeddings_umap[:, 1], c = self.hdbscan_labels, s = 10)
        plt.title("Clusters of Text Chunks (UMAP + HDBSCAN)")
        plt.show()


class ChannelSimilarity:
    def __init__(self):
        self.videos_classified = pd.read_csv("../data/videos_classified.csv")
        self.channel_embeddings = self.videos_classified.groupby(["channel", "ideology"])["embedding"].apply(lambda embeddings: np.mean(np.vstack(embeddings.values), axis=0)).reset_index()

    def compute_cosine_similarity(self):
        results = []
        for i, j in combinations(range(len(self.channel_embeddings)), 2):
            vec_i = self.channel_embeddings.loc[i, "embedding"]
            vec_j = self.channel_embeddings.loc[j, "embedding"]
            cos_sim = np.dot(vec_i, vec_j)
            results.append({"channel1": self.channel_embeddings.loc[i, "channel"], "channel2": self.channel_embeddings.loc[j, "channel"], "cosine_similarity": cos_sim})
        self.channels_similarity_df = pd.DataFrame(results).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)

    def visualize_graph(self):
        channel_ideology = self.videos_classified.groupby("channel")["ideology"].first().to_dict()
        graph = nx.from_pandas_edgelist(df = self.channels_similarity_df, source = "channel1", target = "channel2", edge_attr = "cosine_similarity")
        node_colors = ["skyblue" if channel_ideology[node] == "progressive" else "orangered" for node in graph.nodes()]

        plt.figure(figsize = (10, 7))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_edges(graph, pos, width = [d["cosine_similarity"] * 5 for _, _, d in graph.edges(data = True)], alpha = 0.5)
        nx.draw_networkx_nodes(graph, pos, node_size = 800, node_color = node_colors, edgecolors = "black")
        nx.draw_networkx_labels(graph, pos, font_size = 9)
        plt.show()