from ..dataset.processor.blip_processors import BlipImageEvalProcessor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import logging
import numpy as np
from PIL import Image
import re
import json

import transformers
from ..util.globals import *
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_edit_results_demo)
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..util.runningstats import Covariance, tally

from matplotlib import pyplot as plt
from datasets import load_dataset
from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalTracer:
    """Multimodal causal trace for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = TRACE_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if hparams.model_name == "blip2":
                from ..trainer.blip2_models import Blip2OPT
                
                model = Blip2OPT(
                    vit_model="eva_clip_g",
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    opt_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    qformer_checkpoint=hparams.qformer_checkpoint,
                    cache_dir=hparams.cache_dir
                )
                self.prompt = "Question: {} Short answer:"
                # self.prompt = "{}"
                # Get vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            elif hparams.model_name == "minigpt4":
                from ..trainer.minigpt4_models import MiniGPT4
                # prompt_template = '###Human: {} ###Assistant: ' # For multi-modal input
                # prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {} ASSISTANT:"
                prompt_template = 'USER: {} ASSISTANT:' # For multi-modal input
                # prompt_template="{}" # For pure text input
                end_sym = "###"
                model = MiniGPT4(
                    vit_model="eva_clip_g",
                    q_former_model=hparams.qformer_checkpoint,
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    prompt_template=prompt_template,
                    end_sym=end_sym,
                    llama_model=hparams.name,
                    vit_ckpt=hparams.state_dict_file,
                    pretrained_ckpt=hparams.pretrained_ckpt,
                    cache_dir=hparams.cache_dir,
                )
                # self.prompt = "<Img> <ImageHere> </Img> [vqa] Based on the image, respond to this question with a short answer: {}"     
                # self.prompt = "<Img> <ImageHere> </Img> [vqa] Based on the image, answer the question with a single word: {}"
                # self.prompt = "<Img> <ImageHere> </Img> [vqa] {} Answer in a single word."
                self.prompt = "<Img> <ImageHere> </Img>{} Answer in a single word."
                # self.prompt = "<Img> <ImageHere> </Img> [vqa] {} Answer the single object directly."
                # Get vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            elif hparams.model_name == "llava":
                from ..trainer.llava_models import LLavaModel
                from ..trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
                system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
                prompt_template = system + 'USER: {} ASSISTANT:'
                model = LLavaModel(
                    llava_model=hparams.name,
                    prompt_template=prompt_template,
                    device_map="cuda",
                    cache_dir=hparams.cache_dir,
                )
                self.prompt = DEFAULT_IMAGE_TOKEN + "{}"
                # Get vis_processor
                vis_processor = model.image_processor
            self.model = model
            ## ADD: requires_grad = False
            nethook.set_requires_grad(False, self.model)
            self.model.eval()
            self.vis_tok = vis_processor
            # Get tokenizer 
            if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                tok_name = (
                    hparams.tokenizer_name
                    if hparams.tokenizer_name is not None
                    else hparams.name
                )
                tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                    tok_name,
                    cache_dir=hparams.cache_dir
                )            
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token    
                self.tok = tokenizer                         
        else:
            self.model, self.tok = self.model_name
            
        self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image
        self.result_dir = f"{hparams.result_dir}/{hparams.model_name}/causal_trace"
        os.makedirs(f"{self.result_dir}/cases", exist_ok=True)
        os.makedirs(f"{self.result_dir}/pdfs", exist_ok=True)
        os.makedirs(f"{self.result_dir}/save", exist_ok=True)
    
    def trace(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            plot=True,
            plot_list=None,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1
        
        knowledge = self._prepare_knowledge(prompts, targets, image, **kwargs)
        
        noise_level, uniform_noise = self._noise_level(knowledge)
        
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        all_results = []
        for known_id, known in enumerate(knowledge):
            for kind in None, "mlp", "attn":
                kind_suffix = f"_{kind}" if kind else ""
                filename = f"{self.result_dir}/cases/knowledge_{known_id}{kind_suffix}.npz"
                batch = self._prepare_multimodal_edit(known)
                if batch['image'] == None or not hasattr(self.model,"query_tokens"):
                    skip_query = 0
                else:
                    skip_query = self.model.query_tokens.shape[1]
                start = time()
                if not os.path.isfile(filename):
                    result = self.apply_algo(
                        self.model,
                        self.tok,
                        known,
                        batch,
                        noise=noise_level,
                        kind=kind,
                        uniform_noise=uniform_noise,
                        skip_query=skip_query
                        # replace=self.hparams.replace
                    )
                    numpy_result = {
                        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in result.items()
                    }
                    np.savez(filename, **numpy_result)
                else:
                    numpy_result = np.load(filename, allow_pickle=True)
                exec_time = time() - start
                LOG.info(f"Execution {known_id} tracing took {exec_time}")
                
                if not numpy_result["correct_prediction"]:
                    tqdm.write(f"Skipping {known['prompt']}")
                    continue
                ## metric -> locate
                causal_result = self._locate(numpy_result)
                all_results.append(causal_result)
                if 'save_interval' in kwargs:
                    save_interval = kwargs['save_interval']
                else:
                    save_interval = 100
                if known_id % save_interval == 0:
                    torch.save(all_results, f"{self.result_dir}/save/causal{kind_suffix}_{known_id}.pth")
                
                if plot:
                    if plot_list is None:
                        if known_id > 100:
                            continue
                    else:
                        if known_id not in plot_list:
                            continue
                    plot_result = dict(numpy_result)
                    plot_result["kind"] = kind
                    pdfname = f'{self.result_dir}/pdfs/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.png'
                    pdfname_ = f'{self.result_dir}/pdfs/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}_preds.png'
                    self._plot_trace_heatmap(plot_result, savepdf=pdfname, modelname=self.model_name)
                    self._plot_pred_heatmap(plot_result, self.tok, savepdf=pdfname_, modelname=self.model_name)
                    LOG.info(
                        f"{known_id} tracing: {known['prompt']}--{known['subject']}  \n locate {causal_result['causal_layer']}, {causal_result['causal_token']}, {causal_result['causal_coords']}"
                    )
                torch.save(all_results, f"{self.result_dir}/save/causal{kind_suffix}_final.pth")
        return all_results

    def trace_dataset(self,
                     ds: Dataset,
                     plot=True,
                     plot_list=None,
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')

        num_edits = 1
        # num_edits = self.hparams.batch_size
        
        all_results = []
        for known_id, known in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):
            # known = known_['edit_inner']
            if 'subject' not in known:
                subject = self._guess_subject(known['prompt'])
                known.update(
                    {
                        'subject': known['prompt']
                    }
                )
            noise_level, uniform_noise = self._noise_level([known])
            for kind in None, "mlp", "attn":
                kind_suffix = f"_{kind}" if kind else ""
                filename = f"{self.result_dir}/cases/knowledge_{known_id}{kind_suffix}.npz"
                batch = self._prepare_multimodal_edit(known, **kwargs)
                start = time()
                if not os.path.isfile(filename):
                    result = self.apply_algo(
                        self.model,
                        self.tok,
                        known,
                        batch,
                        noise=noise_level, ## noise_level
                        kind=kind,
                        uniform_noise=uniform_noise,
                        skip_query=self.model.query_tokens.shape[1]
                        # replace=self.hparams.replace
                    )
                    numpy_result = {
                        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in result.items()
                    }
                    np.savez(filename, **numpy_result)
                else:
                    numpy_result = np.load(filename, allow_pickle=True)
                exec_time = time() - start
                LOG.info(f"Execution {known_id} tracing took {exec_time}")
                
                if not numpy_result["correct_prediction"]:
                    tqdm.write(f"Skipping {known['prompt']}")
                    continue
                ## metric -> locate
                causal_result = self._locate(numpy_result)
                all_results.append(causal_result)
                if 'save_interval' in kwargs:
                    save_interval = kwargs['save_interval']
                else:
                    save_interval = 100
                if known_id % save_interval == 0:
                    torch.save(all_results, f"{self.result_dir}/save/causal{kind_suffix}_{known_id}.pth")

                if plot:
                    if plot_list is None:
                        if known_id > 100:
                            continue
                    else:
                        if known_id not in plot_list:
                            continue
                    plot_result = dict(numpy_result)
                    plot_result["kind"] = kind
                    pdfname = f'{self.result_dir}/pdfs/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.png'
                    self._plot_trace_heatmap(plot_result, savepdf=pdfname)
                    LOG.info(
                        f"{known_id} tracing: {known['prompt']}--{known['subject']}  \n locate {causal_result['causal_layer']}, {causal_result['causal_token']}, {causal_result['causal_coords']}"
                    )
                torch.save(all_results, f"{self.result_dir}/save/causal{kind_suffix}_final.pth")
        return all_results
    
    def pred_dataset(self,
                    ds: Dataset,
                    result_path='',
                    max_length=30,
                    stop_token='\n',
                    **kwargs
                    ):
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')
        if self.model.eos_token_id is None:
            self.model.eos_token_id = self.opt_tokenizer(stop_token, add_special_tokens=False).input_ids[0]
        
        ann_sum = 0
        for known_id, known in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):
            # known = known_['edit_inner']
            batch = self._prepare_multimodal_edit(known, **kwargs)
            # self.model.eos_token_id = self.model.opt_tokenizer.eos_token_id
            output_text = self.model.generate(batch, max_length=max_length)
            assert len(output_text) == 1
            ann_sum +=  ds.annotation[known_id]['pred'].lower() == output_text[0].lower()
            print(f"Pre: {ds.annotation[known_id]['pred']}, Post: {output_text[0]}")
            ds.annotation[known_id]['pred'] = output_text[0]
        with open(result_path, "w") as json_file:
            json.dump(ds.annotation, json_file, indent=4)  # `indent` makes the JSON file readable
        print(f"{self.model_name} Annotation['pred']'s accuracy: {ann_sum/len(ds.annotation)}")
        LOG.info(f"{self.model_name} Annotation['pred']'s accuracy: {ann_sum/len(ds.annotation)}")
        
    def _post_process_output(self, output_text, stop_token):
        stop_index = output_text.find(stop_token)
        return output_text[:stop_index] if stop_index != -1 else output_text

    def _prepare_knowledge(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in image]
        image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in image_path]
        if 'llava' in self.hparams.model_name:
            image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(self.hparams.device) if i is not None else None for i in image]
        else:
            image = [self.vis_tok(i).to(self.hparams.device) if i is not None else None for i in image]
        
        knowledge = [{
            'ori_prompt': prompt,
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]
        
        if 'subjects' in kwargs:
            if isinstance(kwargs['subjects'], str):
                kwargs['subjects'] = [kwargs['subjects'],]
            else:
                assert len(kwargs['subjects']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subjects']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, known in enumerate(knowledge):
                known.update(
                    {
                        'subject': kwargs['subjects'][i]
                    }
                )
        else:
            for known in knowledge:
                ## TODO: guess subject？
                subject = self._guess_subject(known['prompt'])
                known.update(
                    {
                        'subject': known['prompt']
                    }
                )
            
        return knowledge

    def _noise_level(self, knowns):
        noise_level = self.hparams.noise_level
        uniform_noise = False
        if isinstance(noise_level, str):
            if noise_level.startswith("s"):
                # Automatic spherical gaussian
                factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
                noise_level = factor * self._collect_embedding_std(
                    self.model, self.tok, [k["subject"] for k in knowns]
                )
                LOG.info(f"Using noise_level {noise_level} to match model times {factor}")
            elif noise_level == "m":
                # Automatic multivariate gaussian
                noise_level = self._collect_embedding_gaussian(self.model, self.tok)
                LOG.info(f"Using multivariate gaussian to match model noise")
            elif noise_level.startswith("t"):
                # Automatic d-distribution with d degrees of freedom
                degrees = float(noise_level[1:])
                noise_level = self._collect_embedding_tdist(self.model, self.tok, degrees)
            elif noise_level.startswith("u"):
                uniform_noise = True
                noise_level = float(noise_level[1:])
        return noise_level, uniform_noise
        
    
    def _prepare_multimodal_edit(self, known, **kwargs):
        target = known['target']
        if 'ori_prompt' in known:
            ori_prompt = known['ori_prompt']
        else:
            ori_prompt = known['prompt']
        prompt = known['prompt']
        image = known['image']
        if 'is_ds' in kwargs and kwargs['is_ds']:
            prompt= self.prompt.format(prompt) if image is not None else prompt
            image = image.to(self.hparams.device) if image is not None else None
        if isinstance(target, str):
            target = [target, ]
        if isinstance(prompt, str):
            prompt = [prompt, ]
        if isinstance(ori_prompt, str):
            ori_prompt = [ori_prompt, ]
        if image is not None and len(image.shape) == 3:
            image = image.unsqueeze(0)
        # text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompt, target)]
        text_input = prompt
        
        if self.hparams.model_name == 'minigpt4':
            prompts_len = [len(self.tok.encode(prompt_, add_special_tokens=False)) for prompt_ in prompt]
            # target = self.tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]
        else:
            prompts_len = [len(self.tok.encode(prompt_, add_special_tokens=False)) for prompt_ in prompt]
            # target = self.tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False,
            #             return_tensors="pt", )["input_ids"]

        ret = {
            'ori_text_input': ori_prompt,
            'text_input': text_input,
            'image': image,
            'answer': target,
            'prompts_len': prompts_len
        }
        return ret
    
    @staticmethod
    def _locate(result, topk=10):
        differences = result["scores"]
        low_score = result["low_score"]
        diff_matrix = np.abs(differences - low_score)
        flat_scores, flat_indices = torch.topk(torch.tensor(diff_matrix).flatten(), topk)
        rows, cols = torch.div(flat_indices, diff_matrix.shape[1], rounding_mode='floor'), flat_indices % diff_matrix.shape[1]
        top_k_coords = list(zip(rows.tolist(), cols.tolist()))
        return dict(
            causal_token=rows.tolist(),
            causal_layer=cols.tolist(),
            causal_coords=top_k_coords,
            causal_score=flat_scores.tolist(),
            low_score=low_score,
            input_tokens=result['input_tokens'],
            subject_range=result['subject_range'],
        )
        # return rows.tolist(), cols.tolist(), top_10_coords
    @staticmethod
    def _plot_pred_heatmap(result, tokenizer, savepdf=None, title=None, xlabel=None, modelname=None):
        preds = result['preds']
        unique_tokens = np.unique(preds)

        token_texts = {token: tokenizer.decode([token]) for token in unique_tokens}

        num_tokens = len(unique_tokens)
        cmap = plt.cm.get_cmap('viridis', num_tokens)

        token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
        indexed_preds = np.vectorize(token_to_index.get)(preds)

        plt.figure(figsize=(3.5, 2), dpi=300)
        plt.imshow(indexed_preds, cmap=cmap, aspect='auto')
        
        nrows, ncols = preds.shape
        for i in range(nrows + 1):
            plt.axhline(i - 0.5, color='black', linewidth=0.5)  # Horizontal lines
        for j in range(ncols + 1):
            plt.axvline(j - 0.5, color='black', linewidth=0.5)  # Vertical lines

        cbar = plt.colorbar(ticks=np.arange(num_tokens), orientation='vertical')
        cbar.ax.set_yticklabels([token_texts[token] for token in unique_tokens], fontsize=12)  # Add token labels

        # Add titles and labels
        if title:
            plt.title(title, fontsize=14)
        if xlabel:
            plt.xlabel(xlabel, fontsize=12)
        if modelname:
            plt.ylabel(f"Model: {modelname}", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Save or show the plot
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
        differences = result["scores"]
        low_score = result["low_score"]
        answer = result["answer"]
        kind = (
            None
            if (not result["kind"] or result["kind"] == "None")
            else str(result["kind"])
        )
        window = result.get("window", 10)
        labels = list(result["input_tokens"])
        for i in range(*result["subject_range"]):
            labels[i] = labels[i] + "*"

        with plt.rc_context(rc={"font.family": "Times New Roman"}):
            fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
            h = ax.pcolor(
                np.abs(differences-low_score),
                cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                    kind
                ],
                vmin=0,
            )
            ax.invert_yaxis()
            ax.set_yticks([0.5 + i for i in range(len(differences))])
            ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
            ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
            ax.set_yticklabels(labels)
            if not modelname:
                modelname = "GPT"
            if not kind:
                ax.set_title("Impact of restoring state after corrupted input")
                ax.set_xlabel(f"single restored layer within {modelname}")
            else:
                kindname = "MLP" if kind == "mlp" else "Attn"
                ax.set_title(f"Impact of restoring {kindname} after corrupted input")
                ax.set_xlabel(f"center of interval of {window} restored {kindname} layers in {modelname}")
            cb = plt.colorbar(h)
            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            elif answer is not None:
                # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
                cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
            if savepdf:
                os.makedirs(os.path.dirname(savepdf), exist_ok=True)
                plt.savefig(savepdf, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

    # @staticmethod
    # def _make_inputs(tokenizer, prompts, device="cuda"):
    #     token_lists = [tokenizer.encode(p) for p in prompts]
    #     maxlen = max(len(t) for t in token_lists)
    #     if "[PAD]" in tokenizer.all_special_tokens:
    #         pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    #     else:
    #         pad_id = 0
    #     input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    #     # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    #     attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    #     return dict(
    #         input_ids=torch.tensor(input_ids).to(device),
    #         #    position_ids=torch.tensor(position_ids).to(device),
    #         attention_mask=torch.tensor(attention_mask).to(device),
    #     )
    
    @staticmethod    
    def _layername(model, num, kind=None):
        if hasattr(model, "transformer"):
            if kind == "embed":
                return "transformer.wte"
            return f'transformer.h.{num}{"" if kind is None else "." + kind}'
        if hasattr(model, "gpt_neox"):
            if kind == "embed":
                return "gpt_neox.embed_in"
            if kind == "attn":
                kind = "attention"
            return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
        if hasattr(model, "opt_model"):
            if kind == "embed":
                return "opt_model.model.decoder.embed_tokens"
            if kind == "attn":
                kind = "self_attn"
            return f'opt_model.model.decoder.layers.{num}{"" if kind is None else "." + kind}'
        if hasattr(model, "llama_model"):
            if kind == "embed":
                return "llama_model.model.embed_tokens"
            if kind == "attn":
                kind = "self_attn"
            return f'llama_model.model.layers.{num}{"" if kind is None else "." + kind}'
        if hasattr(model, "llava_model"):
            if kind == "embed":
                return "llava_model.model.embed_tokens"
            if kind == "attn":
                kind = "self_attn"
            return f'llava_model.model.layers.{num}{"" if kind is None else "." + kind}'
        assert False, "unknown transformer structure"

    def _collect_embedding_std(self, model, tokenizer, subjects):
        alldata = []
        for s in subjects:
            # inp = self._make_inputs(tokenizer, [s])
            with nethook.Trace(model, self._layername(model, 0, "embed")) as t:
                model(dict({"text_input": [s], "image": None, "prompts_len": None, "answer": None, "noise": True}))
                alldata.append(t.output[0])
        alldata = torch.cat(alldata)
        noise_level = alldata.std().item()
        return noise_level

    def _get_embedding_cov(self, model, tokenizer):
        
        def get_ds():
            ds_name = "wikitext"
            raw_ds = load_dataset(
                ds_name,
                dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
            )
            try:
                maxlen = model.config.n_positions
            except:
                maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
            return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

        ds = get_ds()
        sample_size = 1000
        batch_size = 5
        filename = None
        batch_tokens = 100

        progress = lambda x, **k: x

        stat = Covariance()
        loader = tally(
            stat,
            ds,
            cache=filename,
            sample_size=sample_size,
            batch_size=batch_size,
            collate_fn=length_collation(batch_tokens),
            pin_memory=True,
            random_sample=1,
            num_workers=0,
        )
        with torch.no_grad():
            for batch_group in loader:
                for batch in batch_group:
                    batch = dict_to_(batch, "cuda")
                    del batch["position_ids"]
                    with nethook.Trace(model, self._layername(model, 0, "embed")) as tr:
                        model(**batch)
                    feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                    stat.add(feats.cpu().double())
        return stat.mean(), stat.covariance()

    @staticmethod
    def _make_generator_transform(mean=None, cov=None):
        d = len(mean) if mean is not None else len(cov)
        device = mean.device if mean is not None else cov.device
        layer = torch.nn.Linear(d, d, dtype=torch.double)
        nethook.set_requires_grad(False, layer)
        layer.to(device)
        layer.bias[...] = 0 if mean is None else mean
        if cov is None:
            layer.weight[...] = torch.eye(d).to(device)
        else:
            _, s, v = cov.svd()
            w = s.sqrt()[None, :] * v
            layer.weight[...] = w
        return layer


    def _collect_embedding_gaussian(self, model, tokenizer):
        m, c = self._get_embedding_cov(model, tokenizer)
        return self._make_generator_transform(m, c)


    def _collect_embedding_tdist(self, model, tokenizer, degree=3):
        # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
        # And this will give us variance is (degree / degree - 2) * cov.
        # Therefore if we want to match the sample variance, we should
        # reduce cov by a factor of (degree - 2) / degree.
        # In other words we should be sampling sqrt(degree - 2 / u) * sample.
        u_sample = torch.from_numpy(
            np.random.RandomState(2).chisquare(df=degree, size=1000)
        )
        fixed_sample = ((degree - 2) / u_sample).sqrt()
        mvg = self._collect_embedding_gaussian(model, tokenizer)

        def normal_to_student(x):
            gauss = mvg(x)
            size = gauss.shape[:-1].numel()
            factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
            student = factor * gauss
            return student

        return normal_to_student
    
    @staticmethod
    def _guess_subject(prompt):
        return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
            0
        ].strip()
