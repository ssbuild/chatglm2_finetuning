# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 10:11
from config.constant_map import (train_model_config,
                                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)


# 默认禁用lora 相关模块 , lora 和 adalora 只能同时启用一个
lora_info_args = {
    'with_lora': True,  # 是否启用lora模块
    'lora_type': 'lora',
    'r': 8,
    'target_modules': TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[train_model_config['model_type']],
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'fan_in_fan_out': False,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
    'layers_to_transform': None,
    'layers_pattern': None,
    'rank_pattern': {
        # "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
        # "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
    },
    'alpha_pattern': {
        # "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
        # "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
    },

}

adalora_info_args = {
    'with_lora': False,  # 是否启用adalora模块
    'lora_type': 'adalora',
    'r': 8,
    'target_modules': TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[train_model_config['model_type']],
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'fan_in_fan_out': False,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
    'layers_to_transform': None,
    'layers_pattern': None,
    'alpha_pattern': {
        # "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
        # "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
    },

    'target_r':8, # Target Lora matrix dimension.
    'init_r': 12, #Intial Lora matrix dimension.
    'tinit': 0, #The steps of initial warmup.
    'tfinal': 0, #The steps of final warmup.
    'deltaT': 1, #Step interval of rank allocation.
    'beta1': 0.85, #Hyperparameter of EMA.
    'beta2': 0.85, #Hyperparameter of EMA.
    'orth_reg_weight': 0.5, #The orthogonal regularization coefficient.
    'total_step': None, #The total training steps.
    'rank_pattern': None, #The saved rank pattern.
}


ia3_info_args = {
    'with_lora': False,  # 是否启用lora模块
    'target_modules': TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[train_model_config['model_type']],
    'feedforward_modules': TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[train_model_config['model_type']],
    'fan_in_fan_out': False,
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
    'init_ia3_weights': True,
}


train_info_args = {
    'devices': 1,
    'data_backend': 'parquet',  #one of record lmdb arrow_stream ,arrow_file, parquet , 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大
    # 预训练模型路径 , 从0训练，则置空
    **train_model_config,

    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/finetune_train_examples.json'],
    'max_epochs': 20,
    'max_steps': -1,
    'optimizer': 'lion', # one of [lamb,adma,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit]

    'scheduler_type': 'CAWR', #one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]
    'scheduler':{'T_mult': 1,
                 'rewarm_epoch_num': 0.5,  # 如果 max_epochs is not None !
                 # 'T_0': 50000,    # 如果 max_epochs is None , 设定步数
                 'verbose': False},

    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': True},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},


    'optimizer_betas': (0.9, 0.999),
    'train_batch_size': 4,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 2e-4,  #
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 1024, # 如果资源充足，推荐长度2048 与官方保持一致
    'max_target_length': 100,  # 预测最大长度, 保留字段
    'use_fast_tokenizer': False,
    'do_lower_case': None,
    "dataloader_drop_last": True,
    "dataloader_pin_memory": True,
    "dataloader_num_workers": 0,

    ##############  lora模块
    'lora': lora_info_args,
    'adalora': adalora_info_args,
    'ia3': ia3_info_args,
}












train_info_args_hf = {
    'data_backend': 'parquet',
    # one of record lmdb arrow_stream arrow_file,parquet, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大
    # 预训练模型配置
    **train_model_config,

    "output_dir": "./outputs_hf",
    "overwrite_output_dir": True,
    "num_train_epochs": 20,
    "max_steps": -1,
    "save_safetensors": False,
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit":  10,
    "seed": 42,
    "fp16": True,
    'do_train': True,
    'train_file':  [ './data/finetune_train_examples.json'],
    'do_eval': False,
    'do_predict': False,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "evaluation_strategy": "no",
    "eval_steps": 100,
    "optim": "adamw_torch",
        "lr_scheduler_type": "cosine", # one of linear,cosine,cosine_with_restarts,polynomial,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau
    "torch_compile": False,
    "learning_rate": 2e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "weight_decay": 0.,
    "warmup_ratio": 0.03,
    "logging_strategy": "steps",
    "logging_steps": 10,
    "tf32": False,
    "gradient_checkpointing": True,
    'max_seq_length': 512,  #
    'max_target_length': 100,  # 预测最大长度, 保留字段
    'use_fast_tokenizer': False,
    # 'do_lower_case': False,
    "dataloader_drop_last": True,
    "dataloader_pin_memory": True,
    "dataloader_num_workers": 0,

    "log_level": "info",  # 'info', 'warning', 'error' and 'critical , passive',
    ##############  lora模块
    'lora': lora_info_args,
    'adalora': adalora_info_args,
    'ia3': ia3_info_args,

}







