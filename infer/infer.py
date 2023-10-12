# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.chatglm2.llm_model import MyTransformer,ChatGLMTokenizer,PetlArguments,setup_model_profile, ChatGLMConfig
from aigc_zoo.model_zoo.chatglm2.llm_model import RotaryNtkScaledArguments,RotaryLinearScaledArguments # aigc-zoo 0.1.20


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    enable_ntk = False
    rope_args = None
    if enable_ntk:
        #！注意 如果使用 chatglm2-6b-32k 权重 ， 则不用再使用 rope_args
        rope_args = RotaryNtkScaledArguments(model_type='chatglm2',name='rotary_pos_emb',max_position_embeddings=2048, alpha=4)  # 扩展 8k
        # rope_args = RotaryLinearScaledArguments(model_type='chatglm2',name='rotary_pos_emb',max_position_embeddings=2048, scale=4) # 扩展 8k

    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,rope_args=rope_args)

    model = pl_model.get_llm_model()
    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
        #保存量化权重
        # model.save_pretrained('chatglm2-6b-int4',max_shard_size="4GB")
        # exit(0)
    else:
        # 已经量化
        model.half().cuda()
    model = model.eval()

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[], max_length=2048,
                                       eos_token_id=config.eos_token_id,
                                       do_sample=True, top_p=0.7, temperature=0.95, )
        print("input", input)
        print("response", response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)

