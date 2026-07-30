import torch
import torch.nn as nn

from ..constants import IMAGE_TOKEN_INDEX
from .llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from transformers.utils import ModelOutput
from transformers import AutoTokenizer 
from dataclasses import dataclass

from typing import Optional, Tuple, List

@dataclass
class LLavaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    input_tokens: Optional[torch.FloatTensor] = None
    text_input_range: List[tuple] = None
    subject_range: List[tuple] = None
    attention_mask: Optional[torch.FloatTensor] = None
    position_ids: Optional[torch.FloatTensor] = None
    attn_weights: Optional[torch.FloatTensor] = None
    
    
    
class LLavaModel(nn.Module):
    
    def __init__(
        self,
        llava_model="",
        prompt_template="",
        device_map = "cuda",
        max_context_len=3800,
        cache_dir=None,
        vision_tower=None,
        ):
        super().__init__()
        
        self.llava_tokenizer = AutoTokenizer.from_pretrained(llava_model, cache_dir=cache_dir, use_fast=False)
        llava_config = LlavaConfig.from_pretrained(llava_model, cache_dir=cache_dir)
        if vision_tower:
            llava_config.mm_vision_tower = vision_tower
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(
            llava_model,
            config=llava_config,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        print(self.llava_model.hf_device_map.copy())
        self.prompt_template = prompt_template
        vision_tower = self.llava_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.bfloat16)
        self.image_processor = vision_tower.image_processor
        self.max_context_len = max_context_len
    def _device(self):
        return list(self.parameters())[-1].device
    
    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        for pro in prompt:
            prompt_chunks = [self.llava_tokenizer(chunk).input_ids for chunk in pro.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.llava_tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def forward(self, samples, output_attentions=False):
        subject_range = text_input_range = input_tokens = [None]
        if "text_input" in samples:
            if "noise" in samples and samples["noise"]:
                ## only suject embedding with no image and no answer: noise generation for causal tracing, thus no need for prompt template.
                texts = samples['text_input']
            else:
                texts = [self.prompt_template.format(item) for item in samples['text_input']]
        else:
            texts = None
        
        if 'image' in samples and samples['image'] is not None:
            images = [image.to(list(self.parameters())[-1].device) if image is not None else None for image in samples["image"]]
        else:
            images = None
        
        input_ids = []
        for text in texts:
            input_ids.append(self.tokenizer_image_token([text], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(list(self.parameters())[-1].device))
        # input_ids = torch.cat(input_ids, dim=0) 
        id_lens = [input_id.shape[1] for input_id in input_ids]
        pad_ids = torch.tensor(self.llava_tokenizer.pad_token_id, device=list(self.parameters())[-1].device)

        max_length = max(id_lens) if max(id_lens) < self.max_context_len else self.max_context_len
        wrapped_input_ids = pad_ids.expand(len(id_lens), max_length).clone()
        
        # Creating attention_mask based on id_lens
        attention_mask = torch.zeros(len(id_lens), max_length, device=pad_ids.device)
        position_ids = torch.zeros(len(id_lens), max_length, device=pad_ids.device)
        for i, input_id in enumerate(input_ids):
            length = id_lens[i] if id_lens[i] < self.max_context_len else self.max_context_len
            wrapped_input_ids[i, :length] = input_id[0,:length]
            attention_mask[i, :length] = 1 
            position_ids[i, :length] = torch.arange(length, device=pad_ids.device)
        
        input_ids = wrapped_input_ids
        labels = samples['labels'] if 'labels' in samples else None
        if 'trace' in samples and samples['trace']:
            assert 'subject' in samples and 'ori_text_input' in samples, "Causal tracing must specify `subject` and `ori_text_input`."
            subject_ids = self.llava_tokenizer(
                samples['subject'], return_tensors="pt", add_special_tokens=False).input_ids.to(list(self.parameters())[-1].device)
            text_input_ids = self.llava_tokenizer(
                samples['ori_text_input'], padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids.to(list(self.parameters())[-1].device)
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                input_tokens,
                text_input_range,
                subject_range
            ) = self.llava_model.prepare_inputs_labels_for_multimodal_for_trace(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=images,
                subject_ids=subject_ids,
                text_input_ids=text_input_ids)
        else:
            if images is not None and images[0] is not None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.llava_model.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=labels,
                    images=images
                )

            else:
                # if hasattr(self.llava_model, 'base_model'):
                #     inputs_embeds = self.llava_model.base_model.model.embed_tokens(input_ids)
                # else:
                inputs_embeds = self.llava_model.model.embed_tokens(input_ids)
                input_ids = past_key_values = labels = None
        
        outputs = self.llava_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                images=images,
                use_cache=True,
                output_attentions=output_attentions)
        
        return LLavaOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            input_tokens=input_tokens,
            text_input_range=text_input_range,
            subject_range=subject_range,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attn_weights=outputs.attentions if outputs.attentions else None
        )
    
    def generate(
        self, 
        samples, 
        **kwargs,
        ):
        if "text_input" in samples:
            if "noise" in samples and samples["noise"]:
                ## only suject embedding with no image and no answer: noise generation for causal tracing, thus no need for prompt template.
                texts = samples['text_input']
            else:
                texts = [self.prompt_template.format(item) for item in samples['text_input']]
        else:
            texts = None
        
        if 'image' in samples and samples['image'] is not None:
            images = samples["image"]
        else:
            images = None
        
        input_ids = []
        for text in texts:
            input_ids.append(self.tokenizer_image_token([text], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(list(self.parameters())[-1].device))
        # input_ids = torch.cat(input_ids, dim=0) 
        id_lens = [input_id.shape[1] for input_id in input_ids]
        pad_ids = torch.tensor(self.llava_tokenizer.pad_token_id, device=list(self.parameters())[-1].device)

        max_length = max(id_lens) if max(id_lens) < self.max_context_len else self.max_context_len
        wrapped_input_ids = pad_ids.expand(len(id_lens), max_length).clone()
        
        for i, input_id in enumerate(input_ids):
            length = id_lens[i] if id_lens[i] < self.max_context_len else self.max_context_len
            wrapped_input_ids[i, :length] = input_id[:length]
        input_ids = wrapped_input_ids
        outputs = self.llava_model.generate(
                inputs=input_ids,                
                images=images,
                **kwargs)
        
        # answers = []
        # for output_token in outputs:
        #     if output_token[0] == 0:
        #         output_token = output_token[1:]
        #     output_texts = self.llava_tokenizer.decode(output_token, skip_special_tokens=True)
        #     output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
        #     output_texts = output_texts.replace("<s>", "")
        #     output_texts = output_texts.split(r'[/INST]')[-1].strip()
        #     answers.append(output_texts)

        return outputs
    def generate_tokens(
        self, 
        samples, 
        **kwargs,
        ):
        if "text_input" in samples:
            if "noise" in samples and samples["noise"]:
                ## only suject embedding with no image and no answer: noise generation for causal tracing, thus no need for prompt template.
                texts = samples['text_input']
            else:
                texts = [self.prompt_template.format(item) for item in samples['text_input']]
        else:
            texts = None
        
        if 'image' in samples and samples['image'] is not None:
            images = samples["image"]
        else:
            images = None
        
        input_ids = []
        for text in texts:
            input_ids.append(self.tokenizer_image_token([text], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(list(self.parameters())[-1].device))
        # input_ids = torch.cat(input_ids, dim=0) 
        id_lens = [input_id.shape[1] for input_id in input_ids]
        pad_ids = torch.tensor(self.llava_tokenizer.pad_token_id, device=list(self.parameters())[-1].device)

        max_length = max(id_lens) if max(id_lens) < self.max_context_len else self.max_context_len
        wrapped_input_ids = pad_ids.expand(len(id_lens), max_length).clone()
        
        for i, input_id in enumerate(input_ids):
            length = id_lens[i] if id_lens[i] < self.max_context_len else self.max_context_len
            wrapped_input_ids[i, :length] = input_id[:length]
        input_ids = wrapped_input_ids
        
        outputs = self.llava_model.generate(
                inputs=input_ids,                
                images=images,
                **kwargs)
        

        return outputs
