import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import AutoTokenizer 
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from PIL import Image

from transformers import (Qwen2_5_VLForConditionalGeneration, 
                          Qwen2_5_VLProcessor, 
                          AutoProcessor,
                          )
from qwen_vl_utils import process_vision_info


@dataclass
class QwenOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    input_tokens: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.FloatTensor] = None
    position_ids: Optional[torch.FloatTensor] = None
    attn_weights: Optional[torch.FloatTensor] = None
    labels: Optional[torch.FloatTensor] = None
    prompt_lengths: Optional[List[int]] = None
    text_input_range: Optional[List[Tuple[int, int]]] = None
    subject_range: Optional[List[Tuple[int, int]]] = None
    

def _find_last_subsequence(sequence: List[int], pattern: List[int]) -> Optional[Tuple[int, int]]:
    if not pattern or len(pattern) > len(sequence):
        return None
    match = None
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if sequence[start:start + width] == pattern:
            match = (start, start + width)
    return match


def _tensor_image_to_pil(image):
    if not torch.is_tensor(image):
        return image
    image = image.detach().cpu().float().clamp(0, 1)
    if image.dim() == 4 and image.shape[0] == 1:
        image = image[0]
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = (image.numpy() * 255).astype("uint8")
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)
    
    
class QwenVLModel(nn.Module):
    
    def __init__(
        self,
        qwen_model="",
        device_map = "cuda",
        max_context_len=3800,
        cache_dir=None,
        ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model, cache_dir=cache_dir, use_fast=False)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            qwen_model, cache_dir=cache_dir,
            min_pixels=4*28*28, max_pixels=256*28*28  # Limit image resolution to avoid OOM on MMKE
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.max_context_len = max_context_len
    def _device(self):
        return list(self.parameters())[-1].device

    def forward(self, samples, output_attentions=False, prompt_template=True):
        image = samples["image"]
        prompts = samples["text_input"]
        if isinstance(prompts, str):
            prompts = [prompts]
        targets = samples["answer"] if "answer" in samples else [None] * len(prompts)
        if isinstance(targets, str):
            targets = [targets]
        has_targets = any(target is not None for target in targets)
        if isinstance(image, List):
            num_images = len(image)
            if image[0] is None:
                image = None
            else:
                image = [_tensor_image_to_pil(img) for img in image]
        else:
            num_images = 1
            image = _tensor_image_to_pil(image)

        def _user_turn(prompt):
            content = [{"type": "text", "text": prompt}]
            if image is not None:
                content = [{"type": "image"}] + content
            return {"role": "user", "content": content}

        messages = []
        for prompt, target in zip(prompts, targets):
            cur_message = [_user_turn(prompt)]
            if target is not None:
                cur_message.append({"role": "assistant", "content": target})
            messages.append(cur_message)
    
        if prompt_template:
            # do not append the target in the end in generation
            text_input = [self.processor.apply_chat_template(message,
                        add_generation_prompt=not has_targets,
                        tokenize=False) for message in messages]

        else:
            text_input = [
                            {

                                "role": "user",
                                "content": [
                                {"type": "image"}
                            ] * num_images + [{"type": "text", "text": p}],
                        } for p in prompts
            ]
        
        multimodal_inputs = self.processor(
            images=image, 
            text=text_input, 
            return_tensors="pt",
            padding=True).to(self._device(), dtype=torch.bfloat16)
        
        multimodal_inputs.input_ids[multimodal_inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id

        labels = None
        prompt_lengths = [int(x) for x in multimodal_inputs.attention_mask.sum(dim=1).detach().cpu().tolist()]
        if has_targets:
            labels = multimodal_inputs.input_ids.clone()
            prompt_messages = [[_user_turn(prompt)] for prompt in prompts]
            prompt_part = [
                self.processor.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for message in prompt_messages
            ]
            only_prompt_inputs = self.processor(
                text=prompt_part,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self._device(), dtype=torch.bfloat16)
            only_prompt_inputs.input_ids[only_prompt_inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id
            prompt_lengths = [int(x) for x in only_prompt_inputs.attention_mask.sum(dim=1).detach().cpu().tolist()]
            for row, prompt_len in enumerate(prompt_lengths):
                labels[row, :prompt_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
         
        model_kwargs = dict(multimodal_inputs)
        if labels is not None:
            model_kwargs["labels"] = labels
        outputs = self.qwen_model(
                **model_kwargs,
                use_cache=True,
                output_attentions=output_attentions)

        position_ids = (multimodal_inputs.attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        subject_range = None
        text_input_range = None
        if samples.get("trace", False):
            subjects = samples.get("subject", prompts)
            if isinstance(subjects, str):
                subjects = [subjects]
            subject_range = self._locate_text_ranges(multimodal_inputs.input_ids, multimodal_inputs.attention_mask, subjects)
            text_input_range = subject_range
        
        return QwenOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            input_tokens=multimodal_inputs.input_ids,
            attention_mask=multimodal_inputs.attention_mask,
            position_ids=position_ids,
            attn_weights=outputs.attentions if output_attentions and outputs.attentions is not None else None,
            labels=labels,
            prompt_lengths=prompt_lengths,
            text_input_range=text_input_range,
            subject_range=subject_range,
        )

    def _locate_text_ranges(self, input_ids, attention_mask, texts: List[str]) -> List[Tuple[int, int]]:
        ranges = []
        tokenizer = self.processor.tokenizer
        for row, text in enumerate(texts):
            valid_len = int(attention_mask[row].sum().item())
            sequence = input_ids[row, :valid_len].detach().cpu().tolist()
            candidates = [text, text.strip(), f" {text.strip()}"]
            found = None
            for candidate in candidates:
                if not candidate:
                    continue
                pattern = tokenizer.encode(candidate, add_special_tokens=False)
                found = _find_last_subsequence(sequence, pattern)
                if found is not None:
                    break
            if found is None:
                ranges.append((max(valid_len - 1, 0), valid_len))
            else:
                ranges.append(found)
        return ranges
    
    def generate(
        self, 
        samples, 
        **kwargs,
        ):
        image = samples["image"]
        prompts = samples["text_input"]
        targets = samples["answer"]
        if isinstance(image, List):
            num_images = len(image)
        else:
            num_images = 1
        # do not append the target in the end in generation
        text_input = [self.processor.apply_chat_template([
                        {

                            "role": "user",
                            "content": [
                                {"type": "image"}
                            ] * num_images + [{"type": "text", "text": p}],
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False)
                for p, l in zip(prompts, targets)] 
        multimodal_inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self._device(), dtype=torch.bfloat16)

        outputs = self.qwen_model.generate(**multimodal_inputs, **kwargs)
        input_token_length = multimodal_inputs["input_ids"].shape[1]
        outputs = outputs[:,input_token_length:]
        # answers = []
        # for output_token in outputs:
        #     if output_token[0] == 0:
        #         output_token = output_token[1:]
        #     #TODO
            # output_texts = self.tokenizer.decode(output_token, skip_special_tokens=True)
            # output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
            # output_texts = output_texts.replace("<s>", "")
            # output_texts = output_texts.split(r'[/INST]')[-1].strip()
            # answers.append(output_texts)

        return outputs
    def generate_tokens(
        self, 
        samples, 
        **kwargs,
        ):
        image = samples["image"]
        prompts = samples["text_input"]
        targets = samples["answer"]
        if isinstance(image, List):
            num_images = len(image)
        else:
            num_images = 1
            
        # do not append the target in the end in generation
        text_input = [self.processor.apply_chat_template([
                        {

                            "role": "user",
                            "content": [
                                {"type": "image"}
                            ] * num_images + [{"type": "text", "text": p}],
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False)
                for p, l in zip(prompts, targets)] 
        multimodal_inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self._device(), dtype=torch.bfloat16)

        outputs = self.qwen_model.generate(**multimodal_inputs, **kwargs)
        input_token_length = multimodal_inputs["input_ids"].shape[1]
        outputs = outputs[:,input_token_length:]

        return outputs
