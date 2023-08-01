# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps



train_info_models = {
    'chatglm2-6b': {
        'model_type': 'chatglm2',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b',
        'config_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b',
    },

    'chatglm2-6b-int4': {
        'model_type': 'chatglm2',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-int4',
        'config_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-int4',
    },

    'chatglm2-6b-32k': {
        'model_type': 'chatglm2',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-32k',
        'config_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-32k/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-32k',
    },

    'chatglm2-6b-32k-int4': {
        'model_type': 'chatglm2',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-32k-int4',
        'config_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-32k-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-32k-int4',
    },


}


# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    'moss': ['qkv_proj'],
    'chatglm': ['query_key_value'],
    'chatglm2': ['query_key_value'],
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}
