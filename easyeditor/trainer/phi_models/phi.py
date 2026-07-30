import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM
from PIL import Image

# disable flash-attn
# import os 
# from typing import List, Union
# from unittest.mock import patch
# from transformers.dynamic_module_utils import get_imports
# def fixed_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
#     """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
#     if not str(filename).endswith("/modeling_phi.py"):
#         return get_imports(filename)
#     imports = get_imports(filename)
#     imports.remove("flash_attn_2")
#     return imports

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

# 1. 为 Phi-3 Vision 定义一个与 QwenOutput 结构相同的输出类
@dataclass
class Phi3VOutput(ModelOutput):
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
    
    
# 2. 创建 Phi-3 Vision 的外部包装类
class Phi3VLModel(nn.Module):
    
    def __init__(
        self,
        phi3_model_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device_map: str = "cuda",
        cache_dir: Optional[str] = None,
        **kwargs,
        ):
        super().__init__()
        
        # 使用 AutoModelForCausalLM 加载模型，它会自动处理多模态架构
        # with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            phi3_model_name,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True, # Phi-3 需要信任远程代码
            torch_dtype=torch.bfloat16  # <-- Pass the torch.dtype object directly
        )
            
        # config = AutoConfig.from_pretrained(phi3_model_name, cache_dir=cache_dir, trust_remote_code=True)

        # # 2. 强制修改配置
        # config.attn_implementation = "sdpa"

        # # 3. 使用修改后的配置加载模型
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     phi3_model_name,
        #     config=config, # <-- 明确传入修改后的config
        #     trust_remote_code=True, # 仍然需要信任远程代码来构建模型结构
        #     torch_dtype=torch.bfloat16
        # )
        # 使用 AutoProcessor，它统一处理了图像和文本
        self.processor = AutoProcessor.from_pretrained(
            phi3_model_name,
            trust_remote_code=True
        )
        # 确保 tokenizer 有 pad_token，对于批处理至关重要
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _device(self):
        return self.phi_model.device

    def forward(self, samples: Dict[str, Any], output_attentions: bool = False) -> Phi3VOutput:
        # phi3.5 does not support multiple prompts
        if samples["image"] is not None:
            images = samples["image"][0]
        else:
            images = None
        prompts = samples["text_input"][0]
        targets = samples["answer"][0] if "answer" in samples else None
        has_target = targets is not None
        
        if isinstance(images, List):
            num_images = len(images)
        elif images is None:
            num_images = 0
        else:
            num_images = 1
        messages = []
        if num_images == 0:
            messages = [
                    {"role": "user", "content": f"{prompts}"},
                ]
        else:
            messages = [
                        {"role": "user", "content": f"<|image_1|>\n{prompts}"},
                    ]
        if has_target:
            messages.append({"role": "assistant", "content": targets})
    
        text_inputs = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=not has_target, tokenize=False
            )

        inputs = self.processor(
            text=text_inputs,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self._device())
        inputs.input_ids[inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id

        labels = inputs["input_ids"].clone() if has_target else None
        
        # 寻找 assistant turn 的起始位置来 mask 掉 prompt
        # 对于 Phi-3，我们可以寻找 <|assistant|> 之后的 token
        # 一个简化的方法是：只计算 target 的 token 长度，然后从后往前保留
        # 一个更稳健的方法是找到 assistant 标记
        # 找到 assistant 回答的起始位置
        prompt_part = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image_1|>\n{prompts}"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_len = len(self.processor.tokenizer(prompt_part, add_special_tokens=True).input_ids)
        
        # Mask 掉 prompt 部分
        if labels is not None:
            labels[0,:prompt_len] = -100

            # Mask 掉 padding tokens
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # 执行模型的前向传播，并传入 labels 以计算 loss
        # outputs = self.phi_model(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     pixel_values=inputs.pixel_values,
        #     labels=labels,
        #     output_attentions=output_attentions,
        #     use_cache=False,
        # )
        outputs = self.phi_model(
            **inputs,
            labels=labels,
            output_attentions=output_attentions,
            use_cache=True,
        )
        
        return Phi3VOutput(
            loss=outputs.loss,
            logits=outputs.logits
        )
    
    def generate(self, samples: Dict[str, Any], **kwargs) -> List[str]:
        """
        生成文本回答
        """
        outputs = self.generate_tokens(samples, **kwargs)
        
        # 解码生成的 tokens
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 清理输出，移除可能残留的模板标记
        cleaned_responses = []
        for res in responses:
            # Phi-3 的 assistant block 之后的内容才是真正的回答
            # 这是一个简单的后处理示例，可能需要根据具体输出进行调整
            if '<|end|>' in res:
                 res = res.split('<|end|>')[0]
            cleaned_responses.append(res.strip())

        return cleaned_responses

    def generate_tokens(self, samples: Dict[str, Any], **kwargs) -> torch.Tensor:
        if samples["image"] is not None:
            images = samples["image"][0]
        else:
            images = None
        prompts = samples["text_input"][0]

        messages = [{"role": "user", "content": f"<|image_1|>\n{prompts}"}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self._device())
        
        outputs = self.phi_model.generate(**inputs, **kwargs)

        input_token_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_token_length:]

        return new_tokens
# 1. 为 Phi-3 Vision 定义一个与 QwenOutput 结构相同的输出类
@dataclass
class Phi4VOutput(ModelOutput):
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
    
    
# 2. 创建 Phi-3 Vision 的外部包装类
class Phi4VLModel(nn.Module):
    
    def __init__(
        self,
        phi3_model_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device_map: str = "cuda",
        cache_dir: Optional[str] = None,
        **kwargs,
        ):
        super().__init__()
        
        # 使用 AutoModelForCausalLM 加载模型，它会自动处理多模态架构
        # with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            phi3_model_name,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True, # Phi-3 需要信任远程代码
            torch_dtype=torch.bfloat16  # <-- Pass the torch.dtype object directly
        )
        print(self.phi_model.hf_device_map.copy())
            
        # config = AutoConfig.from_pretrained(phi3_model_name, cache_dir=cache_dir, trust_remote_code=True)

        # # 2. 强制修改配置
        # config.attn_implementation = "sdpa"

        # # 3. 使用修改后的配置加载模型
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     phi3_model_name,
        #     config=config, # <-- 明确传入修改后的config
        #     trust_remote_code=True, # 仍然需要信任远程代码来构建模型结构
        #     torch_dtype=torch.bfloat16
        # )
        # 使用 AutoProcessor，它统一处理了图像和文本
        self.processor = AutoProcessor.from_pretrained(
            phi3_model_name,
            trust_remote_code=True
        )
        # 确保 tokenizer 有 pad_token，对于批处理至关重要
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _device(self):
        return self.phi_model.device

    def forward(self, samples: Dict[str, Any], output_attentions: bool = False, freeze_partial_params: bool = False, peft_params = None) -> Phi3VOutput:
        # phi3.5 does not support multiple prompts
        if samples["image"] is not None:
            if isinstance(samples["image"], List):
                images = _tensor_image_to_pil(samples["image"][0])
            else:
                images = _tensor_image_to_pil(samples["image"])
        else:
            images = None
        prompts = samples["text_input"]
        if isinstance(prompts, str):
            prompts = [prompts]
        prompt = prompts[0]
        targets = samples["answer"] if "answer" in samples else [None] * len(prompts)
        if isinstance(targets, str):
            targets = [targets]
        target = targets[0] if targets else None
        has_target = target is not None
        
        if images is None:
            messages = [
                    {"role": "user", "content": f"{prompt}"},
                ]
        else:
            messages = [
                        {"role": "user", "content": f"<|image_1|>\n{prompt}"},
                    ]
        if has_target:
            messages.append({"role": "assistant", "content": target})
    
        text_inputs = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=not has_target, tokenize=False
            )

        inputs = self.processor(
            text=text_inputs,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self._device())
        inputs.input_ids[inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id

        labels = None
        prompt_lengths = [int(x) for x in inputs.attention_mask.sum(dim=1).detach().cpu().tolist()]
        
        if has_target:
            labels = inputs["input_ids"].clone()
            prompt_messages = messages[:1]
            prompt_part = self.processor.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            only_prompt_inputs = self.processor(
                text=prompt_part,
                images=images,
                return_tensors="pt",
                padding=True
            )
            only_prompt_inputs.input_ids[only_prompt_inputs.input_ids == -1] = self.processor.tokenizer.pad_token_id
            prompt_lengths = [int(x) for x in only_prompt_inputs.attention_mask.sum(dim=1).detach().cpu().tolist()]
            for row, prompt_len in enumerate(prompt_lengths):
                labels[row, :prompt_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        model_kwargs = dict(inputs)
        if labels is not None:
            model_kwargs["labels"] = labels
        outputs = self.phi_model(
            **model_kwargs,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if freeze_partial_params:
            for n, p in self.phi_model.named_parameters():
                ## freeze BLinaer
                # and "BLinear" not in n 
                if "ALinear" not in n and p.requires_grad:
                    p.requires_grad = False
                if ("PALinear" in n or "PBLinear"in n )and p.requires_grad:
                    p.requires_grad = False
                if peft_params:
                    if any(peft_param in n for peft_param in peft_params):
                        if "base_layer.base_layer" not in n:
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                        # p.requires_grad = True


        
        position_ids = (inputs.attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        subject_range = None
        text_input_range = None
        if samples.get("trace", False):
            subjects = samples.get("subject", prompts)
            if isinstance(subjects, str):
                subjects = [subjects]
            subject_range = self._locate_text_ranges(inputs.input_ids, inputs.attention_mask, subjects)
            text_input_range = subject_range

        return Phi4VOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            input_tokens=inputs.input_ids,
            attention_mask=inputs.attention_mask,
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
    
    def generate(self, samples: Dict[str, Any], **kwargs) -> List[str]:
        """
        生成文本回答
        """
        outputs = self.generate_tokens(samples, **kwargs)
        
        # 解码生成的 tokens
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 清理输出，移除可能残留的模板标记
        cleaned_responses = []
        for res in responses:
            # Phi-3 的 assistant block 之后的内容才是真正的回答
            # 这是一个简单的后处理示例，可能需要根据具体输出进行调整
            if '<|end|>' in res:
                 res = res.split('<|end|>')[0]
            cleaned_responses.append(res.strip())

        return cleaned_responses

    def generate_tokens(self, samples: Dict[str, Any], **kwargs) -> torch.Tensor:
        if samples["image"] is not None:
            if isinstance(samples["image"],List):
                images = samples["image"][0]
            else:
                images = samples["image"]     
        else:
            images = None
        prompts = samples["text_input"][0]

        if images == None:
            messages = [{"role": "user", "content": f"{prompts}"}]
        else:
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompts}"}]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self._device())
        
        outputs = self.phi_model.generate(**inputs, num_logits_to_keep = 0, num_beams=1, **kwargs)

        input_token_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_token_length:]

        return new_tokens
