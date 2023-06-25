# coding=utf8
# @Time    : 2023/5/12 20:20
# @Author  : tk
# @FileName: chatglm_model

import copy
import os
import re
import warnings
from typing import List, Tuple, Optional, Callable
import torch
from deep_training.nlp.models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration,ChatGLMConfig, logger,setup_model_profile
from deep_training.nlp.models.transformer import TransformerBase
from torch import nn
from transformers import LogitsProcessorList, LogitsProcessor, GenerationConfig, StoppingCriteriaList
from models.tokenization_chatglm import ChatGLMTokenizer


def build_masks_and_position_ids_glm(batch_input_ids, ctxlens, max_len = None):
    if max_len is None:
        max_len = batch_input_ids.size(1)

    batch_position_ids, batch_attention_mask = [], []
    for input_ids, context_length in zip(batch_input_ids, ctxlens):
        if context_length.dim() == 1:
            context_length = context_length.squeeze(dim=-1)

        mask_position = context_length - 1
        position_ids = list(range(context_length)) + [mask_position] * (max_len - context_length)
        block_position_ids = [0] * context_length + list(range(1, max_len - context_length + 1))

        attention_mask = torch.ones((1, max_len, max_len))
        attention_mask = torch.tril(attention_mask)
        attention_mask[..., :context_length] = 1
        attention_mask = (attention_mask < 0.5)

        batch_position_ids.append(torch.stack((torch.tensor(position_ids), torch.tensor(block_position_ids))))
        batch_attention_mask.append(attention_mask)

    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_position_ids = torch.stack(batch_position_ids, dim=0)
    return batch_attention_mask,batch_position_ids



class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self,config):
        super(MyChatGLMForConditionalGeneration, self).__init__(config)


    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:]
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    @torch.no_grad()
    def generate_for_continue_writing(self,tokenizer, query: str, max_length: int = 2048, num_beams=1,
        do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs
    ):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        tokenizer: ChatGLMTokenizer
        inputs_ids = tokenizer.encode(query)
        inputs_ids = torch.tensor(inputs_ids[:-2] + inputs_ids[:-2],dtype=torch.int32).unsqueeze(0)
        attention_mask,position_ids = build_masks_and_position_ids_glm(inputs_ids,[1])
        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        position_ids = position_ids.to(self.device)
        outputs = self.generate(inputs_ids=inputs_ids,attention_mask=attention_mask,position_ids=position_ids, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs_ids[0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        return response

    @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history



class MyTransformerChatGlmLMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        if not load_in_4bit:
            quantization_config = kwargs.get("quantization_config", None)
            if quantization_config:
                load_in_4bit = quantization_config.load_in_4bit

        if not load_in_8bit and not load_in_4bit:
            kwargs.pop("device_map", None)
            kwargs.pop("quantization_config", None)
        super(MyTransformerChatGlmLMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGeneration, *args, **kwargs))

        # for param in self.model.parameters():
        #     param.requires_grad = False  # freeze the model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)

        # class CastOutputToFloat(nn.Sequential):
        #     def forward(self, x):
        #         return super().forward(x).to(torch.float32)
        #
        # self.model.lm_head = CastOutputToFloat(self.model.lm_head)


    def enable_input_require_grads(self):
        pass
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        # self.model.enable_input_require_grads()

