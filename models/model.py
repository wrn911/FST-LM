# -*- coding: utf-8 -*-
"""
时序预测模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SimpleTransformerModel(nn.Module):
    """简单的Transformer时序预测模型"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model

        # 输入投影层
        self.input_projection = nn.Linear(1, self.d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=args.seq_len)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=args.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.n_layers
        )

        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model // 2, args.pred_len)
        )

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 (batch_size, seq_len)

        Returns:
            output: 预测序列 (batch_size, pred_len)
        """
        batch_size, seq_len = x.shape

        # 添加特征维度: (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)

        # 输入投影: (batch_size, seq_len, d_model)
        x = self.input_projection(x)

        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Dropout
        x = self.dropout(x)

        # Transformer编码
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # 使用最后一个时间步的表示进行预测
        last_hidden = encoded[:, -1, :]  # (batch_size, d_model)

        # 输出投影
        output = self.output_projection(last_hidden)  # (batch_size, pred_len)

        return output


class SimpleLSTMModel(nn.Module):
    """简单的LSTM时序预测模型"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.d_model

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=args.n_layers,
            dropout=args.dropout if args.n_layers > 1 else 0,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_dim // 2, args.pred_len)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 (batch_size, seq_len)

        Returns:
            output: 预测序列 (batch_size, pred_len)
        """
        batch_size, seq_len = x.shape

        # 添加特征维度: (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # 输出预测
        output = self.output_layer(last_output)  # (batch_size, pred_len)

        return output


class SimpleMLPModel(nn.Module):
    """简单的MLP时序预测模型（基线模型）"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(args.seq_len, args.d_model * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model * 2, args.d_model),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model, args.pred_len)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 (batch_size, seq_len)

        Returns:
            output: 预测序列 (batch_size, pred_len)
        """
        # 直接通过MLP
        output = self.mlp(x)  # (batch_size, pred_len)

        return output


def create_model(args, model_type='transformer'):
    """
    创建模型的工厂函数

    Args:
        args: 配置参数
        model_type: 模型类型 ('transformer', 'lstm', 'mlp')

    Returns:
        model: 创建的模型
    """
    if model_type == 'transformer':
        model = SimpleTransformerModel(args)
    elif model_type == 'lstm':
        model = SimpleLSTMModel(args)
    elif model_type == 'mlp':
        model = SimpleMLPModel(args)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, args):
    """打印模型信息"""
    num_params = count_parameters(model)
    print(f"模型类型: {model.__class__.__name__}")
    print(f"参数数量: {num_params:,}")
    print(f"模型大小: {num_params * 4 / 1024 / 1024:.2f} MB")
    print(f"输入形状: (batch_size, {args.seq_len})")
    print(f"输出形状: (batch_size, {args.pred_len})")