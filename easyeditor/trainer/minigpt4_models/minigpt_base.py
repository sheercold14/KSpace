import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn


from .base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils import ModelOutput
from typing import Optional, Tuple, List
from dataclasses import dataclass
# from minigpt4.conversation.conversation import StoppingCriteriaSub

@dataclass
class MiniGPTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.IntTensor = None
    attention_mask: torch.IntTensor = None
    input_lens: List[int] = None
    input_tokens: Optional[torch.FloatTensor] = None
    text_input_range: List[tuple] = None
    subject_range: List[tuple] = None
    
class MiniGPTBase(BaseModel):
    """
    Base class for MiniGPT-4 and MiniGPT-v2
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llama_model="",
        max_txt_len=32,
        max_context_len=3800,
        prompt_template="",
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,  # lora_r means lora is not used
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        vit_ckpt=None,
        cache_dir=None,
    ):
        super().__init__()

        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model,
            low_resource=low_resource,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            cache_dir=cache_dir,
        )

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit, vit_ckpt
        )

        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template
        self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt
        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
                pn = each_img_embed.shape[-2]
                if lengths is not None:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                    each_img_embed = each_img_embed[:lengths[idx] * pn]
                p_segs = each_prompt.split('<ImageHere>')
                interleave_emb = []
                for idx, seg in enumerate(p_segs[:-1]):
                    p_tokens = self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    p_embed = self.embed_tokens(p_tokens.input_ids) 
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                wrapped_emb = torch.cat(interleave_emb, dim=1)
                p_tokens = self.llama_tokenizer(
                    p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))

            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)
            
            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
            return wrapped_embs, wrapped_atts
    
    def prompt_wrap_for_trace(self, img_embeds, atts_img, prompts, lengths=None, subjects=None, text_inputs=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img, [None]*len(img_embeds), [None]*len(img_embeds), [None]*len(img_embeds)
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt, prompt_tokens, [(0, prompt_embeds.shape[1])]*len(prompt_embeds), [None]*len(prompt_embeds)
        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)
            
            text_input_range = []
            subject_range = []
            
            p_tokens = self.llama_tokenizer(
                    prompts, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_embed = self.embed_tokens(p_tokens.input_ids)
            
            for idx, (each_img_embed, each_subject, each_text_input) in enumerate(zip(img_embeds, subjects, text_inputs)):
                pn = each_img_embed.shape[-2]
                if lengths is not None:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                    each_img_embed = each_img_embed[:lengths[idx] * pn]
                
                subject_tokens = self.llama_tokenizer(
                    each_subject, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                subject_ids = subject_tokens.input_ids[0].tolist()  # List of token IDs for subject
                subject_start = self._find_subsequence(p_tokens.input_ids[idx].tolist(), subject_ids)
                subject_end = subject_start[0] + len(subject_ids)
                
                image_tokens = self.llama_tokenizer('<ImageHere>', return_tensors="pt", add_special_tokens=False)
                image_tokens_idx = self._find_subsequence(p_tokens.input_ids[idx].tolist(), image_tokens.input_ids[0].tolist())
                
                wrapped_emb = []
                wrapped_tokens= []
                last_idx = 0
                for img_idx, token_idx in enumerate(image_tokens_idx):
                    wrapped_emb.append(p_embed[idx, last_idx: token_idx])
                    wrapped_tokens.extend(p_tokens.input_ids[idx, last_idx:token_idx].tolist())
                    start = img_idx * pn
                    end = (img_idx + 1) * pn
                    if start < each_img_embed.shape[0]:  
                        wrapped_emb.append(each_img_embed[start:end])
                        wrapped_tokens.extend([0]*pn)
                    last_idx = token_idx + len(image_tokens.input_ids[0])
                wrapped_emb.append(p_embed[idx, last_idx:]) 
                wrapped_tokens.extend(p_tokens.input_ids[idx, last_idx:].tolist())
                
                wrapped_emb = torch.cat(wrapped_emb, dim=0).unsqueeze(0)
                
                if '[vqa]' in each_text_input:
                    each_text_input = each_text_input.split('[vqa]')[-1][1:]
                # if '\n' in each_text_input:
                #     each_text_input = each_text_input.split('\n')[-1]
                text_input_tokens = self.llama_tokenizer(
                    each_text_input, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                text_input_ids = text_input_tokens.input_ids.squeeze().tolist()  # List of token IDs for text_input
                if each_text_input[0] == '\n':
                    text_input_ids = text_input_ids[1:]
                    text_input_tokens['input_ids'] = text_input_tokens['input_ids'][:, 1:]
                    text_input_tokens['attention_mask'] = text_input_tokens['attention_mask'][:, 1:]
                text_input_start = self._find_subsequence(wrapped_tokens, text_input_ids)  # Find the start index
                text_input_end = text_input_start[0] + len(text_input_ids)
                
                emb_lists.append(wrapped_emb)
                text_input_range.append((text_input_start[0], text_input_end))
                subject_range.append((subject_start[0], subject_end))
                

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))

            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)
            
            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
            return wrapped_embs, wrapped_atts, text_input_tokens, text_input_range, subject_range

    @staticmethod
    def _find_subsequence(sequence, subsequence):
        pos = []
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                pos.append(i)
        if len(pos):
            return pos
        else:
            raise ValueError("Subsequence not found in the sequence.")
        
    
    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        if output_embs is None:
            ## only suject embedding with no image and no answer: noise generation for causal tracing.
            input_lens = []
            for i in range(input_embs.size(0)):
                input_len = input_atts[i].sum()
                input_lens.append(input_len)
            return input_embs, input_atts, input_lens
        else:
            input_lens = []
            cat_embs = []
            cat_atts = []
            for i in range(input_embs.size(0)):
                input_len = input_atts[i].sum()
                input_lens.append(input_len)
                cat_embs.append(
                    torch.cat([
                        input_embs[i][:input_len],
                        output_embs[i],
                        input_embs[i][input_len:]
                    ])
                )
                cat_atts.append(
                    torch.cat([
                        input_atts[i][:input_len],
                        output_atts[i],
                        input_atts[i][input_len:]
                    ])
                )
            cat_embs = torch.stack(cat_embs)
            cat_atts = torch.stack(cat_atts)
            return cat_embs, cat_atts, input_lens

    def tokenize_conversation(self, conv_q, conv_a):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        to_regress_token_ids_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]  # the first question is handled in the prompt wrap function, skip it
            answers = [self.llama_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)

            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)

            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.llama_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.pad_token_id).to(torch.int)

        return to_regress_token_ids, to_regress_token_attn, targets

    def preparing_embedding(self, samples):
        ### prepare input tokens
        if 'image' in samples and samples['image'] is not None:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None
        
        regress_embeds = regress_atts = part_targets = subject_range = text_input_range = input_tokens = None
        if 'conv_q' in samples:
            # handeling conversation datasets
            conv_q, conv_a = samples['conv_q'], samples['conv_a']

            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]

            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]

            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)

        else:
            if "text_input" in samples:
                if "noise" in samples:
                    ## only suject embedding with no image and no answer: noise generation for causal tracing, thus no need for prompt template.
                    instruction = samples['text_input']
                else:
                    instruction = [self.prompt_template.format(item) for item in samples['text_input']]
            elif "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                if 'trace' in samples and samples['trace']:
                    assert 'subject' in samples and 'ori_text_input' in samples, "Causal tracing must specify `subject` and `ori_text_input`."
                    cond_embeds, cond_atts, input_tokens, text_input_range, subject_range = self.prompt_wrap_for_trace(img_embeds, img_atts, instruction, subjects=samples['subject'], text_inputs=samples['ori_text_input'])
                else:
                    cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            if samples["answer"] is not None:
                 ## only text_input embedding with no answer: for causal tracing.
                text = [t + self.end_sym for t in samples["answer"]]

                regress_tokens = self.llama_tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False
                ).to(self.device)

                regress_token_ids = regress_tokens.input_ids
                regress_atts = regress_tokens.attention_mask
                part_targets = regress_token_ids.masked_fill(
                    regress_token_ids == self.llama_tokenizer.pad_token_id, -100
                )
        if regress_atts is not None:
            regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, input_tokens, text_input_range, subject_range

    def forward(self, samples, reduction='mean'):
        # prepare the embedding to condition and the embedding to regress
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, input_tokens, text_input_range, subject_range = \
            self.preparing_embedding(samples)

        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        ## only obtain suject embedding: noise generation for causal tracing.
        if not 'noise' in samples:
            # get bos token embedding
            bos = torch.ones_like(cond_atts[:, :1]) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            bos_atts = cond_atts[:, :1]

            # add bos token at the begining
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

            if text_input_range is not None:
                text_input_range = [(a+1, b+1)for (a,b) in text_input_range]
        
        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        
        if part_targets is not None:
            for i, target in enumerate(part_targets):
                targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )
        loss = outputs.loss

        return MiniGPTOutput(
            loss=loss,
            logits=outputs.logits,
            labels=targets,
            attention_mask=attention_mask,
            input_lens=input_lens,
            input_tokens=input_tokens,
            text_input_range=text_input_range,
            subject_range=subject_range
        )

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @torch.no_grad()
    def generate(
        self,
        samples,
        num_beams=1,
        max_length=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        #     stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        if "text_input" in samples:
            texts = [self.prompt_template.format(item) for item in samples['text_input']]
        elif self.prompt_list:
            texts = random.choice(self.prompt_list)
        else:
            texts = None
        
        if 'image' in samples and samples['image'] is not None:
            img_embeds, atts_img = self.encode_img(samples["image"])
        else:
            img_embeds = atts_img = None
        # img_embeds, atts_img = self.encode_img(images.to(self.device))
        # image_lists = [[image_emb[None]] for image_emb in img_embeds]

        if 'trace' in samples and samples['trace']:
            assert 'subject' in samples and 'ori_text_input' in samples, "Causal tracing must specify `subject` and `ori_text_input`."
            batch_embs,_,_,_,_ = self.prompt_wrap_for_trace(img_embeds, atts_img, texts, subjects=samples['subject'], text_inputs=samples['ori_text_input'])
        else:
            # batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]
            batch_embs,_ = self.prompt_wrap(img_embeds, atts_img, texts)

        # if isinstance(batch_embs, list):
        #     batch_size = len(batch_embs)
        #     max_len = max([emb.shape[1] for emb in batch_embs])
        #     emb_dim = batch_embs[0].shape[2]
        #     dtype = batch_embs[0].dtype
        #     device = batch_embs[0].device

        #     embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        #     attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        #     for i, emb in enumerate(batch_embs):
        #         emb_len = emb.shape[1]
        #         embs[i, -emb_len:] = emb[0]
        #         attn_mask[i, -emb_len:] = 1
        
            # bos = torch.ones_like(attn_mask[:, :1]) * self.llama_tokenizer.bos_token_id
            # bos_embeds = self.embed_tokens(bos)
            # bos_atts = attn_mask[:, :1]

            # # add bos token at the begining
            # embs = torch.cat([bos_embeds, embs], dim=1)
            # attn_mask = torch.cat([bos_atts, attn_mask], dim=1)
        # else:
        # dtype = batch_embs[0].dtype
        device = batch_embs[0].device
        attn_mask = torch.ones([batch_embs.shape[0], batch_embs.shape[1]], dtype=torch.int, device=device)
        bos = torch.ones_like(attn_mask[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = attn_mask[:, :1]

        # add bos token at the begining
        batch_embs = torch.cat([bos_embeds, batch_embs], dim=1)
        attn_mask = torch.cat([bos_atts, attn_mask], dim=1)
            
        
        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=batch_embs,
                attention_mask=attn_mask,
                max_new_tokens=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.eos_token_id,
                # stopping_criteria=stopping_criteria,
            )

        # with self.maybe_autocast():
        #     outputs = self.llama_model.generate(
        #         inputs_embeds=embs,
        #         attention_mask=attn_mask,
        #         max_new_tokens=max_new_tokens,
        #         num_beams=num_beams,
        #         do_sample=do_sample,
        #         # stopping_criteria=stopping_criteria,
        #     )
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers

    # @torch.no_grad()
    # def generate(
    #     self,
    #     samples,
    #     num_beams=1,
    #     max_length=20,
    #     min_length=1,
    #     top_p=0.9,
    #     repetition_penalty=1,
    #     length_penalty=1,
    #     temperature=1,
    #     do_sample=False,
    #     stop_words_ids=[2],
    # ):
    #     '''
    #         function for generate test use
    #     '''

    #     # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
    #     #     stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
    #     if 'image' in samples and samples['image'] is not None:
    #         img_embeds, atts_img = self.encode_img(samples["image"])
    #     else:
    #         img_embeds = atts_img = None
        
    #     if "text_input" in samples:
    #         texts = [self.prompt_template.format(item) for item in samples['text_input']]
    #     elif self.prompt_list:
    #         texts = random.choice(self.prompt_list)
    #     else:
    #         texts = None
    #     # img_embeds, atts_img = self.encode_img(images.to(self.device))
    #     image_lists = [[image_emb[None]] for image_emb in img_embeds]

    #     batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

    #     batch_size = len(batch_embs)
    #     max_len = max([emb.shape[1] for emb in batch_embs])
    #     emb_dim = batch_embs[0].shape[2]
    #     dtype = batch_embs[0].dtype
    #     device = batch_embs[0].device

    #     embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
    #     attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
    #     for i, emb in enumerate(batch_embs):
    #         emb_len = emb.shape[1]
    #         embs[i, -emb_len:] = emb[0]
    #         attn_mask[i, -emb_len:] = 1

    #     with self.maybe_autocast():
    #         outputs = self.llama_model.generate(
    #             inputs_embeds=embs,
    #             attention_mask=attn_mask,
    #             max_new_tokens=max_length,
    #             num_beams=num_beams,
    #             length_penalty=length_penalty,
    #             temperature=temperature,
    #             do_sample=do_sample,
    #             min_length=min_length,
    #             top_p=top_p,
    #             repetition_penalty=repetition_penalty,
    #             eos_token_id=self.eos_token_id,
    #             # stopping_criteria=stopping_criteria,
    #         )

    #     # with self.maybe_autocast():
    #     #     outputs = self.llama_model.generate(
    #     #         inputs_embeds=embs,
    #     #         attention_mask=attn_mask,
    #     #         max_new_tokens=max_new_tokens,
    #     #         num_beams=num_beams,
    #     #         do_sample=do_sample,
    #     #         # stopping_criteria=stopping_criteria,
    #     #     )
    #     answers = []
    #     for output_token in outputs:
    #         if output_token[0] == 0:
    #             output_token = output_token[1:]
    #         output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
    #         output_texts = output_texts.split('###')[0]  # remove the stop sign </s>
    #         output_texts = output_texts.replace("<s>", "")
    #         output_texts = output_texts.split(r'[/INST]')[-1].strip()
    #         answers.append(output_texts)

    #     return answers

    @torch.no_grad()
    def multi_select(self, images, texts, answers, num_cand=None):
        all_losses = []
        for answer in answers:
            choice_samples = {
                'image': images,
                'instruction_input': texts,
                'answer': answer
            }
            loss = self.forward(choice_samples, reduction='none')['loss'].reshape(-1, 1)
            all_losses.append(loss)
            torch.cuda.empty_cache()
        all_losses = torch.cat(all_losses, dim=-1)
        if num_cand is not None:
            for i in range(all_losses.shape[0]):
                all_losses[i, num_cand[i]:] = 9999
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return output_class_ranks.tolist()
