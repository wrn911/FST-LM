# -*- coding: utf-8 -*-
"""
FedDA地理位置聚类模块
"""
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import copy


class FedDAGeographicClustering:
    """FedDA地理位置和流量模式聚类"""

    def __init__(self, num_clusters=3, max_iterations=10):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.client_clusters = {}

    def iterative_clustering(self, federated_data: Dict, augmented_data: Dict):
        """
        迭代聚类算法：结合地理位置和增强流量数据

        Args:
            federated_data: 联邦数据，包含坐标信息
            augmented_data: 增强数据

        Returns:
            clusters: 聚类结果 {cluster_id: [client_ids]}
        """
        coordinates = federated_data['coordinates']
        client_ids = list(coordinates.keys())

        # 准备地理位置数据
        geo_data = []
        traffic_data = []
        valid_clients = []

        for client_id in client_ids:
            if client_id in coordinates and client_id in augmented_data:
                coord = coordinates[client_id]
                geo_data.append([coord['lng'], coord['lat']])
                traffic_data.append(augmented_data[client_id])
                valid_clients.append(client_id)

        if len(valid_clients) < self.num_clusters:
            # 客户端数量不足，简单分组
            return self._simple_grouping(valid_clients)

        geo_data = np.array(geo_data)
        traffic_data = np.array(traffic_data)

        # 标准化地理坐标
        geo_data = self._normalize_coordinates(geo_data)

        # 标准化流量数据
        traffic_data = self._normalize_traffic(traffic_data)

        # 迭代聚类
        clusters = self._iterative_kmeans(geo_data, traffic_data, valid_clients)

        return clusters

    def _normalize_coordinates(self, geo_data):
        """标准化地理坐标"""
        mean = np.mean(geo_data, axis=0)
        std = np.std(geo_data, axis=0)
        std[std == 0] = 1  # 避免除零
        return (geo_data - mean) / std

    def _normalize_traffic(self, traffic_data):
        """标准化流量数据"""
        # 如果流量数据是多维的，需要展平
        if len(traffic_data.shape) > 2:
            traffic_data = traffic_data.reshape(traffic_data.shape[0], -1)

        mean = np.mean(traffic_data, axis=0)
        std = np.std(traffic_data, axis=0)
        std[std == 0] = 1
        return (traffic_data - mean) / std

    def _iterative_kmeans(self, geo_data, traffic_data, client_ids):
        """迭代K-means聚类"""
        # 初始化：基于流量数据聚类
        traffic_kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        traffic_labels = traffic_kmeans.fit_predict(traffic_data)

        for iteration in range(self.max_iterations):
            # 基于地理位置聚类
            geo_kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            geo_labels = geo_kmeans.fit_predict(geo_data)

            # 检查是否收敛（两种聚类结果一致）
            if np.array_equal(traffic_labels, geo_labels):
                break

            # 更新流量聚类的中心，使用地理聚类结果
            traffic_centers = []
            for cluster_id in range(self.num_clusters):
                cluster_mask = geo_labels == cluster_id
                if np.any(cluster_mask):
                    center = np.mean(traffic_data[cluster_mask], axis=0)
                    traffic_centers.append(center)
                else:
                    # 空簇，使用随机中心
                    traffic_centers.append(np.random.random(traffic_data.shape[1]))

            # 重新分配流量聚类
            traffic_kmeans.cluster_centers_ = np.array(traffic_centers)
            traffic_labels = traffic_kmeans.predict(traffic_data)

        # 构建最终聚类结果
        clusters = {}
        for cluster_id in range(self.num_clusters):
            clusters[cluster_id] = []

        for i, client_id in enumerate(client_ids):
            cluster_id = int(traffic_labels[i])
            clusters[cluster_id].append(client_id)

        # 记录聚类结果
        for client_id, cluster_id in zip(client_ids, traffic_labels):
            self.client_clusters[client_id] = cluster_id

        return clusters

    def _simple_grouping(self, client_ids):
        """简单分组（当客户端数量不足时）"""
        clusters = {}
        for i, client_id in enumerate(client_ids):
            cluster_id = i % self.num_clusters
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_id)
            self.client_clusters[client_id] = cluster_id

        return clusters

    def get_client_cluster(self, client_id):
        """获取客户端所属聚类"""
        return self.client_clusters.get(client_id, 0)