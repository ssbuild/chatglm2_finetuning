# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import random
import typing
from enum import Enum
import numpy as np
from aigc_zoo.model_zoo.chatglm2.llm_model import ChatGLMTokenizer

class DataStrategy(Enum):
    truncation = 1
    siding = 2






def build_template_chatglm(query, answer = None , prefix=None, history=None):
    prompt = prefix or ''
    sid = 0
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += query if sid == 0 else "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_chatglm2(query, answer = None,prefix=None,  history=None):
    prompt = prefix or ''
    sid = 1
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt


def build_template_default(query, answer = None, prefix=None, history=None):
    prompt = prefix or ''
    if history is not None:
        for q,a in history:
            prompt += "User: {}\nAssistant:{}".format(q,a)
    prompt += "User: {}\nAssistant:".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_tiger(query,answer = None, prefix=None, history=None):
    prompt = prefix or ''
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    if history is not None:
        for q,a in history:
            prompt += "{}{}{}{}".format(tok_ins,q,tok_res,a)

    prompt += "{}{}{}".format(tok_ins, query, tok_res)
    if answer is not None:
        prompt += answer
    return prompt


# 切换模版
build_template = build_template_chatglm2


#对截断
class TokenIdsMaker:
    @classmethod
    def final(cls, input_ids: typing.List, labels, max_seq_length, tokenizer):
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
    @classmethod
    def trunction(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List,sup=True):
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids = tokenizer.encode(text=build_template(q,prefix=prefix, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False)
            while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
                if len(b_ids) > len(a_ids):
                    b_ids.pop(-1)
                else:
                    a_ids.pop(0)
            b_ids += [config.eos_token_id]
            input_ids = a_ids + b_ids
            labels = copy.deepcopy(input_ids) if not sup else [-100] * len(a_ids) + copy.deepcopy(b_ids)
            input_ids = sptoken + input_ids
            labels = sptoken + labels if not sup else [-100] * len(sptoken) + labels
            assert len(input_ids) <= max_seq_length
            ds.append(cls.final(input_ids, labels, max_seq_length, tokenizer))
        return ds


    @classmethod
    def slidding(cls, tokenizer: ChatGLMTokenizer,config, examples,
                 max_seq_length,
                 sptoken: typing.List,
                 sliding_size = None,
                 src_max_length=-1,
                 dst_max_length=-1,
                 sup=True):
        if sliding_size is None or sliding_size < 0:
            sliding_size = max_seq_length - len(sptoken)

        assert sliding_size <= max_seq_length - len(sptoken)

        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids = tokenizer.encode(text=build_template(q,prefix=prefix, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False)
            if src_max_length and src_max_length > 0:
                a_ids = a_ids[:src_max_length]
            if dst_max_length and dst_max_length > 0:
                b_ids = b_ids[:dst_max_length]

            b_ids += [config.eos_token_id]
            input_ids_qa = a_ids + b_ids
            labels_all = copy.deepcopy(input_ids_qa) if not sup else [-100] * len(a_ids) + b_ids

            pos = 0
            while pos < len(input_ids_qa):
                input_ids = input_ids_qa[pos:pos + max_seq_length - len(sptoken)]
                labels = labels_all[pos:pos + max_seq_length - len(sptoken)]

                pos += sliding_size
                if np.all(np.asarray(labels) == -100):
                    continue

                input_ids = sptoken + input_ids
                labels = sptoken + labels if not sup else [-100] * len(sptoken) + labels
                ds.append(cls.final(input_ids,labels,max_seq_length,tokenizer))
        return ds
