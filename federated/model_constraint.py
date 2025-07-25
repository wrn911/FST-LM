import torch
import numpy as np


class ModelParameterConstraint:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.previous_trainable_params = None

    def constrain_model_update(self, current_model_state):
        # 提取可训练参数
        current_trainable = self._extract_trainable_params(current_model_state)

        if self.previous_trainable_params is None:
            self.previous_trainable_params = current_trainable.copy()
            return current_model_state, {'constrained': False}

        # 计算变化幅度
        deviation = self._compute_deviation(current_trainable, self.previous_trainable_params)

        if deviation > self.threshold:
            # 应用约束
            constrained_trainable = self._apply_constraint(
                current_trainable, self.previous_trainable_params, deviation
            )

            # 更新模型
            constrained_model = current_model_state.copy()
            constrained_model.update(constrained_trainable)

            self.previous_trainable_params = constrained_trainable.copy()
            return constrained_model, {'constrained': True, 'deviation': deviation}

        else:
            self.previous_trainable_params = current_trainable.copy()
            return current_model_state, {'constrained': False}

    def _extract_trainable_params(self, model_state):
        trainable_params = {}
        for param_name, param_value in model_state.items():
            if self._is_trainable_param(param_name):
                trainable_params[param_name] = param_value
        return trainable_params

    def _is_trainable_param(self, param_name):
        # LoRA参数
        if 'lora_A' in param_name or 'lora_B' in param_name:
            return True
        # TimeLLM层
        if any(layer in param_name for layer in ['ts2language', 'output_projection', 'normalize_layers']):
            return True
        return False

    def _compute_deviation(self, current_params, previous_params):
        total_squared_diff = 0.0
        for param_name in current_params.keys():
            if param_name in previous_params:
                param_diff = current_params[param_name] - previous_params[param_name]
                total_squared_diff += torch.norm(param_diff).item() ** 2
        return np.sqrt(total_squared_diff)

    def _apply_constraint(self, current_params, previous_params, deviation):
        scale_factor = self.threshold / deviation
        constrained_params = {}

        for param_name in current_params.keys():
            if param_name in previous_params:
                param_diff = current_params[param_name] - previous_params[param_name]
                constrained_diff = scale_factor * param_diff
                constrained_params[param_name] = previous_params[param_name] + constrained_diff
            else:
                constrained_params[param_name] = current_params[param_name]

        return constrained_params