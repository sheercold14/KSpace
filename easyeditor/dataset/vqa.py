"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoProcessor
class SimpleResizeProcessor:
    def __init__(self, size=(336, 336)):
        self.size = size

    def __call__(self, image, return_tensors=None):
        # image: PIL.Image.Image
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image")
        resized = image.resize(self.size, Image.BICUBIC)
        if return_tensors == "pt":
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToTensor(),  # Converts to (C, H, W) float tensor
            ])
            return {"pixel_values": transform(resized)}
        else:
            return resized
class VQADataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        tokenizer = None
        if config.model_name == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_name == "llava":
            vision_tower = getattr(config, "vision_tower", None) or "openai/clip-vit-large-patch14-336"
            vis_processor = transformers.CLIPImageProcessor.from_pretrained(vision_tower)
            # A local CLIP checkpoint can be supplied through the public config.
            
            return_tensors = True
            # Alternatively, use a Hugging Face model identifier.
        elif config.model_name ==  "qwen-vl":
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        elif "qwen2.5_vl" in config.model_name.lower() or "phi3_vl" in config.model_name.lower() or "phi4_vl" in config.model_name.lower():
            #from transformers import Qwen2VLImageProcessor
            #vis_processor = Qwen2VLImageProcessor.from_pretrained(config.name)
            vis_processor = SimpleResizeProcessor(size=(336, 336))
            return_tensors = False
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(config.tokenizer_name, trust_remote_code=True).tokenizer            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token    
        else:
            raise NotImplementedError("unknown model class")
        if tokenizer is None:
            if (config is not None and hasattr(config, 'tokenizer_name')):
                tok_name = (
                    config.tokenizer_name
                    if config.tokenizer_name is not None
                    else config.name
                )
                if config.tokenizer_class == "QWenTokenizer":
                    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
                elif config.model_name == "owl-2":
                    tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
                else:
                    tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                        tok_name, trust_remote_code=True
                    )            
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer:"

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]  
        for i, record in enumerate(tqdm(self.annotation, desc="Processing Records")):
            
            if record['alt'] == "":
                continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            
            image = Image.open(image_path).convert("RGB")
            rephrase_image = Image.open(rephrase_image_path).convert("RGB")
            locality_image = Image.open(locality_image_path).convert("RGB")
            
            ori_image = image
            ori_rephrase_image = rephrase_image
            ori_locality_image = locality_image
            if self.vis_processor is not None:
                if return_tensors:
                    image = self.vis_processor(image, return_tensors="pt")['pixel_values'].to(dtype=torch.float16)
                    rephrase_image = self.vis_processor(rephrase_image, return_tensors="pt")['pixel_values'].to(dtype=torch.float16) 
                    locality_image = self.vis_processor(locality_image, return_tensors="pt")['pixel_values'].to(dtype=torch.float16) 
                else:
                    image = [self.vis_processor(image)]
                    rephrase_image = [self.vis_processor(rephrase_image)]
                    locality_image = [self.vis_processor(locality_image)]
                
            else:
                image = [image]
                rephrase_image = [rephrase_image]
                locality_image = [locality_image]     
            item = {
                'case_id': len(data),
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_rephrase': rephrase_image,
                'ori_image': ori_image,
                'ori_rephrase_image': ori_rephrase_image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image
            item['ori_multimodal_locality_image'] = ori_locality_image
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn_bp(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [" " + b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [" " + b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [" " + b['multimodal_locality_ground_truth'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [self.prompt.format(r) + t for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r), add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r))) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [q + a for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
        
        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
    
    def collate_fn(self, batch):
        elem = batch[0]
        collated = {}

        for key in elem:
            if isinstance(elem[key], torch.Tensor):
                collated[key] = torch.stack([d[key] for d in batch])
            elif isinstance(elem[key], (int, float, str)):
                collated[key] = [d[key] for d in batch]
            elif isinstance(elem[key], Image.Image):
                collated[key] = [d[key] for d in batch]
            elif isinstance(elem[key], list):
                collated[key] = [d[key] for d in batch]
            else:
                raise TypeError(f"Unsupported type in batch for key '{key}': {type(elem[key])}")
        return collated
    
import json
from torchvision import transforms
# To compute cov for ROME, MEMIT、 AlphaEdit
class VQADataset_Simple(BaseDataset):
    def __init__(self, prompt, template, annotation_file, image_root, size=None, image_size=256):
        self.image_root = image_root
        with open(annotation_file,'r',encoding='utf-8') as f:
            if size:
                self.annotations = json.load(f)[:size]
            else: 
                self.annotations = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.prompt = prompt
        self.template = template
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_name = ann["image"]
        txt = ann["src"]
        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image":image.half(),
            "text_input": self.template.format(self.prompt.format(txt)) if self.template else self.prompt.format(txt)
        }
    @staticmethod
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text_input"] for item in batch]
        # image_tensor = torch.stack(images.unsqueeze(0),dim=0)
        return {
            "image":images,
            "text_input":texts
        }
    
    def __len__(self):
        return len(self.annotations)

class VQADataset_X(BaseDataset):
    def __init__(self, annotation_file, image_root, prompt=None, template=None, size=None, image_size=256):
        self.image_root = image_root
        with open(annotation_file,'r',encoding='utf-8') as f:
            if size:
                self.annotations = json.load(f)[:size]
            else: 
                self.annotations = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize(image_size[0]),
            transforms.ToTensor(),
        ])
        self.prompt = prompt
        self.template = template
        self.image_size = image_size
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_name = ann["image"]
        txt = ann["src"]
        img_path = os.path.join(self.image_root, img_name)
        answer = ann["pred"]
        # m_loc_image = ann[]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        self.PIL_processor = SimpleResizeProcessor(size=self.image_size)
        txt = self.prompt.format(txt) if self.prompt else txt
        
        loc_prompt = ann['loc']
        loc_image = None
        loc_answer = ann["loc_ans"]
        return {
            "PIL_image": self.PIL_processor(Image.open(img_path).convert("RGB")),
            "image":image.half(),
            "text_input": self.template.format(txt) if self.template else txt,
            "answer": answer,
            "loc_image": None,
            "loc_prompt": self.template.format(loc_prompt) if self.template else loc_prompt,
            "loc_answer": loc_answer
        }
    @staticmethod
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text_input"] for item in batch]
        answers = [item["answer"] for item in batch]
        # image_tensor = torch.stack(images.unsqueeze(0),dim=0)
        return {
            "image":images,
            "text_input":texts, 
            "answer": answers
        }
    def __len__(self):
        return len(self.annotations)
    


# def get_VQA_ds(hparams, prompt, template, size=None):
#     annotation_path = hparams.train_annotation_path
#     image_root = hparams.coco_image
#     raw_ds = VQADataset_Simple(size=size, prompt=prompt,template=template,annotation_file=annotation_path,image_root=image_root,image_size=336)
#     return raw_ds

# from .coco_caption import COCOCaptionDataset_X
# def get_Caption_ds(hparams, prompt, template, size=None):
#     annotation_path = hparams.caption_train_annotation_path
#     image_root = hparams.coco_image
#     raw_ds = COCOCaptionDataset_X(size=size, prompt=prompt, template=template, annotation_path=annotation_path, image_root=image_root, image_size=336)
#     return raw_ds

