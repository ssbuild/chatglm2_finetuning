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




def build_template_chatglm(query, answer = None, history=None):
    prompt = ''
    sid = 0
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += query if sid == 0 else "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_chatglm2(query, answer = None, history=None):
    prompt = ''
    sid = 1
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt


def build_template_default(query, answer = None, history=None):
    prompt = ''
    if history is not None:
        for q,a in history:
            prompt += "User: {}\nAssistant:{}".format(q,a)
    prompt += "User: {}\nAssistant:".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_tiger(query,answer = None, history=None):
    prompt = ''
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
class TokenTruncation:

    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List,ensure_answer_min_length=1,sup=True):
        assert ensure_answer_min_length > 0
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids, b_ids = [], []
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a)[:max_seq_length - 3 - ensure_answer_min_length] + [config.eos_token_id]

            a_len = max_seq_length - len(b_ids) - 1
            input_ids = a_ids[-a_len:] + b_ids
            if sup:
                labels = [-100] * a_len + input_ids[a_len:]
            else:
                labels = copy.deepcopy(input_ids)
            input_ids = sptoken + input_ids
            labels = [-100] * len(sptoken) + labels
            ds.append(TokenIdsFinal.process(input_ids,labels,max_seq_length,tokenizer))
        return ds

class TokenSiding:
    @classmethod
    def process(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List,sliding_size = None,sup=True):
        if sliding_size is None:
            sliding_size = max_seq_length
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids, b_ids = [], []
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False) + [config.eos_token_id]

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
