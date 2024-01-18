import os.path

import faiss
import multiprocessing
import numpy as np
import json


class FaissNearestNeighbour:
    def __init__(self, num_workers=multiprocessing.cpu_count(), n_nearest_neighbours=1, max_distance=999999, on_gpu=faiss.get_num_gpus() > 0):
        faiss.omp_set_num_threads(num_workers)
        self.search_index = None
        self.n_nearest_neighbours = n_nearest_neighbours
        self.good_n_total = None
        self.on_gpu = on_gpu
        self.max_distance = max_distance

    def train(self, features):
        self.good_n_total = features.shape[0]
        if self.search_index:
            self.reset_index()
        self.search_index = self.create_index(features.shape[-1])
        self.search_index.add(features)

    def train_ng_ex(self, ng_ex_features):
        reconstructed_data = np.zeros((self.good_n_total + ng_ex_features.shape[0], ng_ex_features.shape[-1]),
                                      dtype='float32')
        for i in range(self.good_n_total):
            reconstructed_data[i] = self.search_index.reconstruct(i)
        reconstructed_data[self.good_n_total:, :] = ng_ex_features
        train_good_n_total = self.good_n_total
        self.train(reconstructed_data)
        self.good_n_total = train_good_n_total

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

    def create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())
        return faiss.IndexFlatL2(dimension)

    def save(self, output_file_path):
        search_index = faiss.index_gpu_to_cpu(self.search_index) if self.on_gpu else self.search_index
        faiss.write_index(search_index, output_file_path)

    def load(self, input_faiss_path, good_n_total):
        self.reset_index()
        self.search_index = faiss.read_index(input_faiss_path)
        self.search_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.search_index,
                                                   faiss.GpuClonerOptions()) if self.on_gpu else self.search_index
        self.good_n_total = good_n_total

    def predict(self, features):
        anomaly_scores_list = []
        for feature in features:
            query_distances, query_nns = self.search_index.search(feature, self.n_nearest_neighbours)
            query_distances = np.mean(query_distances, axis=-1)
            anomaly_index_array = np.asarray(
                [index > (self.good_n_total - 1) for index in np.squeeze(query_nns, axis=-1).tolist()])
            query_distances[anomaly_index_array] = self.max_distance
            anomaly_scores_list.append(query_distances)
        anomaly_scores = np.stack(anomaly_scores_list, axis=0)
        return anomaly_scores
