# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import random
import typing
from enum import Enum
import numpy as np
from aigc_zoo.model_zoo.chatglm2.chatglm_model import ChatGLMTokenizer

class DataStrategy(Enum):
    truncation = 1
    siding = 2

class TokenIdsFinal:
    @classmethod
    def process(cls,input_ids: typing.List,labels,max_seq_length,tokenizer):

        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen

        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))

        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen,
        }
        return d


#对prompt 截断

class TokenTruncation:

    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, a_ids, b_ids, max_seq_length, sptoken: typing.List,ensure_answer_min_length=1,sup=True):
        input_ids = a_ids[:max_seq_length - len(b_ids) -3 - ensure_answer_min_length] + b_ids
        a_len = len(input_ids) - len(b_ids)
        input_ids = input_ids[:max_seq_length - 3] + [config.eos_token_id]
        if sup:
            labels = [-100] * a_len + input_ids[a_len:]
        else:
            labels = copy.deepcopy(input_ids)
        input_ids = sptoken + input_ids
        labels = [-100] * len(sptoken) + labels

        d = TokenIdsFinal.process(input_ids,labels,max_seq_length,tokenizer)
        return [d]

class TokenSiding:
    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, a_ids, b_ids, max_seq_length, sptoken: typing.List,sliding_size = None,sup=True):
        if sliding_size is None:
            sliding_size = max_seq_length
        ds = []
        input_ids_qa = a_ids + b_ids + [config.eos_token_id]
        if sup:
            labels_all = [-100] * len(a_ids) + b_ids
        else:
            labels_all = copy.deepcopy(input_ids_qa)

        pos = 0
        while pos < len(input_ids_qa):
            input_ids = sptoken + input_ids_qa[pos:pos + max_seq_length - 2]
            labels = [-100] * len(sptoken) + labels_all[pos:pos + max_seq_length - 2]

            pos += sliding_size
            if np.all(np.asarray(labels) == -100):
                continue
            d = TokenIdsFinal.process(input_ids,labels,max_seq_length,tokenizer)
            ds.append(d)
        return ds
