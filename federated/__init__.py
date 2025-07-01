# -*- coding: utf-8 -*-
"""
联邦学习模块
"""

from .client import FederatedClient
from .server import FederatedServer
from .aggregation import FedAvgAggregator, WeightedFedAvgAggregator, get_aggregator

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'FedAvgAggregator',
    'WeightedFedAvgAggregator',
    'get_aggregator'
]