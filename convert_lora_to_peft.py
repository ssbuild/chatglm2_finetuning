# -*- coding: utf-8 -*-
# @Time:  20:50
# @Author: tk
# @File：convert_lora_to_peft

import os
import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper, global_args
from aigc_zoo.model_zoo.chatglm2.llm_model import MyTransformer, ChatGLMTokenizer, setup_model_profile, ChatGLMConfig, \
    LoraArguments,LoraModel

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args, ) = parser.parse_dict(train_info_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args,)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ckpt_dir = './best_ckpt/last'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)

    lora_args = LoraArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True and config.pre_seq_len is None

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens,#扩充词
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir,is_trainable=True)

    pl_model.eval().half().cuda()

    lora_model: LoraModel = pl_model.backbone # noqa

    lora_model.save_pretrained('./new_lora')



