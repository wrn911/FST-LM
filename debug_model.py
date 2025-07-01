#!/usr/bin/env python3
"""
调试模型结构，查看可用的线性层模块
"""

import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.TimeLLM import Model
from utils.config import get_args


class DebugModelConfig:
    """调试用的简化模型配置"""

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 1
        self.d_model = 64
        self.n_heads = 8
        self.d_ff = 256

        # LLM配置
        self.llm_model = 'GPT2'
        self.llm_dim = 768
        self.llm_layers = 6

        # 补丁配置
        self.patch_len = 16
        self.stride = 8

        self.dropout = 0.1
        self.prompt_domain = True
        self.content = "Debug model"

        # 不使用LoRA进行调试
        self.use_lora = False


def debug_model_structure():
    """调试模型结构"""
    print("=== 调试模型结构 ===")

    try:
        # 创建模型
        config = DebugModelConfig()
        model = Model(config)

        print("\n=== LLM模型结构分析 ===")
        print(f"LLM模型类型: {type(model.llm_model)}")

        # 分析所有模块
        linear_modules = {}
        all_modules = {}

        for name, module in model.llm_model.named_modules():
            module_type = type(module).__name__
            all_modules[name] = module_type

            # 检查是否为线性层
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                linear_modules[name] = {
                    'type': module_type,
                    'in_features': module.weight.shape[1],
                    'out_features': module.weight.shape[0]
                }

        print(f"\n发现 {len(linear_modules)} 个线性层:")
        for name, info in linear_modules.items():
            print(f"  {name}: {info['type']} ({info['in_features']} -> {info['out_features']})")

        # 提取模块名称的最后部分
        module_names = set()
        for name in linear_modules.keys():
            module_name = name.split('.')[-1]
            module_names.add(module_name)

        print(f"\n唯一的模块名称: {sorted(module_names)}")

        # 推荐的LoRA目标模块
        attention_candidates = []
        for name in module_names:
            if any(keyword in name.lower() for keyword in ['attn', 'query', 'key', 'value', 'proj']):
                attention_candidates.append(name)

        print(f"\n推荐的LoRA目标模块: {attention_candidates}")

        return attention_candidates

    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    target_modules = debug_model_structure()

    if target_modules:
        print(f"\n建议的命令行参数:")
        modules_str = ",".join(target_modules)
        print(f"--lora_target_modules {modules_str}")
    else:
        print("\n无法自动检测目标模块，请手动指定")