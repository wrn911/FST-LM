from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

# 添加PEFT支持
try:
    from peft import LoraConfig, get_peft_model, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    print("警告: PEFT库未安装，将禁用LoRA功能。请运行: pip install peft")
    PEFT_AVAILABLE = False

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim  # 这个值会根据选择的LLM模型动态设置
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 选择模型
        if configs.llm_model == 'DeepSeek':
            self._init_deepseek_model(configs)
        elif configs.llm_model == 'GPT2':
            self._init_gpt2_model(configs)
        elif configs.llm_model == 'BERT':
            self._init_bert_model(configs)
        elif configs.llm_model == 'LLAMA':
            self._init_llama_model(configs)
        elif configs.llm_model == 'Qwen3':
            self._init_qwen3_model(configs)
        else:
            raise Exception('LLM model is not defined')

        # 重要：获取实际的LLM隐藏维度
        self.d_llm = self.llm_model.config.hidden_size
        print(f"LLM隐藏维度: {self.d_llm}")

        # 设置tokenizer的pad_token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 添加LoRA支持
        self._apply_lora_if_enabled(configs)

        # 设置描述信息
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The dataset records the wireless traffic of a certain base station'

        self.dropout = nn.Dropout(configs.dropout)

        self.ts2language = Embedding_layer(configs)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # 确保所有新添加的层使用float32
        self._ensure_float32_layers()

    def _init_qwen3_model(self, configs):
        """初始化Qwen3模型 - 修复SSL问题"""
        try:
            from transformers import Qwen2Config, Qwen2Model, Qwen2Tokenizer
        except ImportError:
            print("警告: 请安装最新版transformers以支持Qwen3: pip install transformers>=4.37.0")
            raise

        # 方案1A: 强制使用本地文件
        try:
            # 首先尝试本地加载
            self.qwen_config = Qwen2Config.from_pretrained(
                'Qwen/Qwen3-0.6B',
                local_files_only=True,  # 强制只使用本地文件
                trust_remote_code=True
            )
            print("✓ 成功从本地加载Qwen3配置")
        except Exception as e:
            print(f"本地配置加载失败: {e}")
            print("尝试创建默认配置...")

            # 方案1B: 创建默认配置
            self.qwen_config = Qwen2Config(
                vocab_size=151936,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=configs.llm_layers,
                num_attention_heads=16,
                num_key_value_heads=16,
                max_position_embeddings=32768,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=True
            )
            print("✓ 使用默认Qwen3配置")

        # 更新层数设置
        self.qwen_config.num_hidden_layers = configs.llm_layers
        self.qwen_config.output_attentions = True
        self.qwen_config.output_hidden_states = True

        # 加载模型 - 强制本地
        try:
            self.llm_model = Qwen2Model.from_pretrained(
                'Qwen/Qwen3-0.6B',
                trust_remote_code=True,
                local_files_only=True,  # 强制本地
                config=self.qwen_config,
            )
            print("✓ 成功从本地加载Qwen3模型")
        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            print("请确保模型文件已正确下载到本地")
            raise

        # 加载tokenizer - 强制本地
        try:
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                'Qwen/Qwen3-0.6B',
                trust_remote_code=True,
                local_files_only=True  # 强制本地
            )
            print("✓ 成功从本地加载Qwen3 tokenizer")
        except Exception as e:
            print(f"❌ 本地tokenizer加载失败: {e}")
            print("请确保tokenizer文件已正确下载到本地")
            raise

        # 更新配置中的LLM维度为实际值
        configs.llm_dim = self.llm_model.config.hidden_size
        print(f"Qwen3模型隐藏维度: {configs.llm_dim}")

    def _init_deepseek_model(self, configs):
        """初始化DeepSeek模型"""
        self.deepseek_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        self.deepseek_config.num_hidden_layers = configs.llm_layers
        self.deepseek_config.output_attentions = True
        self.deepseek_config.output_hidden_states = True

        try:
            self.llm_model = AutoModel.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=True,
                config=self.deepseek_config,
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = AutoModel.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=False,
                config=self.deepseek_config,
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                trust_remote_code=True,
                local_files_only=False
            )

    def _init_gpt2_model(self, configs):
        """初始化GPT2模型"""
        self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
        self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True

        try:
            self.llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False
            )

    def _init_bert_model(self, configs):
        """初始化BERT模型"""
        self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        self.bert_config.num_hidden_layers = configs.llm_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        try:
            self.llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=True,
                config=self.bert_config,
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=self.bert_config,
            )

        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False
            )

    def _init_llama_model(self, configs):
        """初始化LLaMA模型"""
        self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        self.llama_config.num_hidden_layers = configs.llm_layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True

        try:
            self.llm_model = LlamaModel.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = LlamaModel.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
                config=self.llama_config,
            )

        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False
            )

    def _apply_lora_if_enabled(self, configs):
        """如果启用LoRA，则应用LoRA配置"""
        if hasattr(configs, 'use_lora') and configs.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("LoRA功能需要安装PEFT库: pip install peft")

            print("正在应用LoRA配置...")

            # 根据模型类型设置目标模块
            target_modules = self._get_target_modules(configs)
            print(f"目标模块: {target_modules}")

            # 特殊处理GPT2的Conv1D层
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=configs.lora_rank,
                lora_alpha=configs.lora_alpha,
                lora_dropout=configs.lora_dropout,
                target_modules=target_modules,
                bias="none",
                fan_in_fan_out=False,  # GPT2需要设置为False
                modules_to_save=None,
            )

            try:
                # 应用LoRA
                self.llm_model = get_peft_model(self.llm_model, lora_config)

                # 打印参数统计
                trainable_params = sum(p.numel() for p in self.llm_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.llm_model.parameters())

                print(f"LoRA配置完成:")
                print(f"  可训练参数: {trainable_params:,}")
                print(f"  总参数: {total_params:,}")
                print(f"  可训练参数比例: {100 * trainable_params / total_params:.2f}%")

            except Exception as e:
                print(f"LoRA配置失败: {e}")
                print("尝试使用更保守的配置...")

                # 使用更保守的配置重试
                conservative_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=min(8, configs.lora_rank),  # 降低rank
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=["c_attn"],  # 只使用一个模块
                    bias="none",
                    fan_in_fan_out=False,
                )

                self.llm_model = get_peft_model(self.llm_model, conservative_config)
                print("使用保守配置成功应用LoRA")

    def _get_target_modules(self, configs):
        """根据模型类型获取目标模块"""
        if hasattr(configs, 'lora_target_modules') and configs.lora_target_modules:
            return configs.lora_target_modules

        # 根据实际模型架构自动检测目标模块
        target_modules = self._auto_detect_target_modules()
        if target_modules:
            return target_modules

        # 如果自动检测失败，使用默认配置
        if configs.llm_model == 'Qwen3':
            return ["q_proj", "k_proj", "v_proj", "o_proj"]  # Qwen系列的注意力层
        elif configs.llm_model == 'GPT2':
            return ["c_attn", "c_proj"] # GPT2的注意力层
        elif configs.llm_model == 'BERT':
            return ["query", "value", "key", "dense"]  # BERT的注意力层
        elif configs.llm_model in ['LLAMA', 'DeepSeek']:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]  # LLaMA系列的注意力层
        else:
            return ["c_attn"]  # 最通用的默认值

    def _auto_detect_target_modules(self):
        """自动检测模型中可用的目标模块"""
        import re

        # 获取所有模块名称
        all_modules = set()
        for name, module in self.llm_model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                # 只考虑线性层
                module_name = name.split('.')[-1]  # 获取最后一部分名称
                all_modules.add(module_name)

        print(f"检测到的线性层模块: {sorted(all_modules)}")

        # 定义各种可能的注意力模块名称
        attention_patterns = [
            # GPT系列
            'c_attn', 'c_proj', 'attn.c_attn', 'attn.c_proj',
            # BERT系列
            'query', 'key', 'value', 'dense',
            # LLaMA/DeepSeek系列
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            # 通用模式
            'self_attn', 'attention'
        ]

        # 找到匹配的模块
        found_modules = []
        for pattern in attention_patterns:
            if pattern in all_modules:
                found_modules.append(pattern)

        # 返回找到的模块，如果太少就返回None让使用默认值
        if len(found_modules) >= 2:
            print(f"自动检测到目标模块: {found_modules}")
            return found_modules
        else:
            print("自动检测的目标模块太少，将使用默认配置")
            return None

    def get_trainable_parameters(self):
        """获取可训练参数的状态字典"""
        if hasattr(self.llm_model, 'peft_config'):
            # 如果使用了PEFT，只返回可训练的参数
            return {k: v for k, v in self.named_parameters() if v.requires_grad}
        else:
            # 否则返回所有参数
            return dict(self.named_parameters())

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        # 时序与语义对齐
        enc_out, n_vars = self.ts2language(x_enc)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def _ensure_float32_layers(self):
        """确保所有非LLM层使用float32"""
        self.output_projection = self.output_projection.float()
        self.normalize_layers = self.normalize_layers.float()
        self.ts2language = self.ts2language.float()

class Embedding_layer(nn.Module):
    def __init__(self, configs):
        super(Embedding_layer, self).__init__()
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, configs.dropout)
        self.output_projection = nn.Linear(configs.d_model, configs.llm_dim, bias=False)

    def forward(self, x_enc):
        enc_out, n_vars = self.patch_embedding(x_enc)
        dec_out = self.output_projection(enc_out)
        return dec_out, n_vars


