from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .batch_editor import BatchEditor
# from ..evaluate import (compute_icl_multimodal_edit_quality, 
#                         compute_multimodal_edit_results,
#                         compute_multimodal_edit_results_demo,
#                         compute_mmke_multimodal_edit_quality)
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
import pprint

from .utils import _chunks
import random
import math

# For multiple portability inputs
from ..evaluate.multimodal_evaluate import prepare_multimodal_edit, compute_multimodal_edit_quality_demo, test_generation_quality
import typing
from typing import Optional, Union, List, Tuple, Dict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditor_UnKE:
    """Multimodal editor for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
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
                self.prompt_template = "{}"
                self.image_toks = 32
                # Get vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            elif hparams.model_name == "minigpt4":
                from ..trainer.minigpt4_models import MiniGPT4
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
                self.prompt = "<Img> <ImageHere> </Img>{} Answer in a single word."
                self.prompt_template = prompt_template
                self.image_toks = 32
                # Get vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            elif hparams.model_name == "llava":
                from ..trainer.llava_models import LLavaModel
                from ..trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
                system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
                prompt_template = system + 'USER: {} ASSISTANT:'
                if isinstance(hparams.device, str):
                    model = LLavaModel(
                    llava_model=hparams.name,
                    prompt_template=prompt_template,
                    device_map="auto",
                    cache_dir=hparams.cache_dir)
                else:
                    model = LLavaModel(
                        llava_model=hparams.name,
                        prompt_template=prompt_template,
                        device_map="cuda",
                        cache_dir=hparams.cache_dir,
                    )
                self.prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
                self.prompt_template = prompt_template
                self.image_toks = 576 - 1
                # Get vis_processor
                vis_processor = model.image_processor
            self.model = model
            self.vis_tok = vis_processor
            if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                tok_name = (
                    hparams.tokenizer_name
                    if hparams.tokenizer_name is not None
                    else hparams.name
                )
                tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                    tok_name
                )            
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token    
                self.tok = tokenizer                         
        else:
            self.model, self.tok = self.model_name
        
        if isinstance(hparams.device, int):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image
    def compute_multimodal_edit_results(self,
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        record: typing.Dict,
        device
    ) -> typing.Dict:
        """
        Given a rewritten model, computes generalization and specificity metrics for
        the desired rewrite (passed in via the CounterFact dataset record). Returns a
        dictionary containing those metrics.

        :param model: Rewritten model
        :param tok: Tokenizer
        :param record: CounterFact dataset record
        :paran snips: ???
        :param vec: ???
        :return: Dictionary containing rewriting metrics
        """
        ret = {}
        # First, unpack rewrite evaluation record.

        target = record["target"]
        rewrite_prompts = record["prompt"]
        if record["image"] is not None:
            image = record["image"] if record["image"].is_cuda else record["image"].to(f"cuda:{hparams.device}")
        else:
            image = record["image"]
        prompt_template = record["prompt_template"] if "prompt_template" in record else "{}"
        
        edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image, prompt_template=prompt_template)
        ret['rewrite_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_inner, tok)
        ret['rewrite_gen'] = test_generation_quality(model, edit_inner)

        if "rephrase_prompt" in record.keys():
            rephrase_prompts = record["rephrase_prompt"]
            edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image, prompt_template=prompt_template)
            ret['rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_outer, tok)
            ret['rephrase_gen'] = test_generation_quality(model, edit_outer)
            
        if "image_rephrase" in record.keys():
            rephrase_image = record["image_rephrase"]
            rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(f"cuda:{hparams.device}")
            edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image, prompt_template=prompt_template)
            ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality_demo(model, edit_image_outer, tok)
            ret['image_rephrase_gen'] = test_generation_quality(model, edit_image_outer)

        if 'locality_prompt' in record.keys():
            locality_prompt = record["locality_prompt"]
            locality_ground_truth = record["locality_ground_truth"]
            locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None, prompt_template=prompt_template)
            ret['locality_acc'], ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality, tok)
            ret['locality_gen'] = test_generation_quality(model, locality)

        if 'multimodal_locality_prompt' in record.keys():
            m_loc_prompt = record["multimodal_locality_prompt"]
            m_loc_ground_truth = record["multimodal_locality_ground_truth"]
            m_loc_image = record["multimodal_locality_image"]
            if m_loc_image is not None:
                m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(f"cuda:{hparams.device}")
            m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image, prompt_template=prompt_template)
            ret['multimodal_locality_acc'], ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, m_locality, tok)
            ret['multimodal_locality_gen'] = test_generation_quality(model, m_locality)

        if 'portability_prompt' in record.keys():
            portability_prompt = record["portability_prompt"]
            portability_ground_truth = record["portability_ground_truth"]
            portability_image = record["portability_image"]
            # if portability_image is not None:
            #     portability_image = portability_image if portability_image.is_cuda else portability_image.to(f"cuda:{hparams.device}")
            portability = []
            for i in range(len(portability_prompt)):
                portability.append(prepare_multimodal_edit(hparams, tok, portability_ground_truth[i], portability_prompt[i], portability_image[i], prompt_template=prompt_template))
                # _, ret['portability_output'] = compute_multimodal_edit_quality_demo(model, portability, tok)
                ret[f'portability_gen_{i}'] = test_generation_quality(model, portability[i])

        if 'multimodal_portability_prompt' in record.keys():
            m_port_prompt = record["multimodal_portability_prompt"]
            m_port_ground_truth = record["multimodal_portability_ground_truth"]
            m_port_image = record["multimodal_portability_image"]
            if m_port_image is not None:
                m_port_image = m_port_image if m_port_image.is_cuda else m_port_image.to(f"cuda:{hparams.device}")
            m_portability = prepare_multimodal_edit(hparams, tok, m_port_ground_truth, m_port_prompt, m_port_image, prompt_template=prompt_template)
            # _, ret['multimodal_portability_output'] = compute_multimodal_edit_quality_demo(model, m_portability, tok)
            ret['multimodal_portability_gen'] = test_generation_quality(model, m_portability)
        # Form a list of lists of prefixes to test.

        return ret
    def _prepare_requests_batch_for_unke(self,
        prompts: Union[str, List[str]],
        targets: Union[str, List[str]],
        image: Union[str, List[str]],
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        rephrase_image: Optional[Union[str, List[str]]] = None,
        locality_inputs: Optional[List[Dict]] = None,
        portability_inputs: Optional[List[Dict]] = None,
        **kwargs):
        # Ensure that inputs are lists if they are not already
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(image, str):
            image = [image]
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts]
        if isinstance(rephrase_image, str):
            rephrase_image = [rephrase_image]
        if isinstance(locality_inputs, dict):
            locality_inputs = [locality_inputs]
        if isinstance(portability_inputs, dict):
            portability_inputs = [portability_inputs]

        # Ensure that all lists have the same length
        assert len(prompts) == len(targets) == len(image), "Prompts, targets, and images must have the same length"

        # Replicate locality_inputs if necessary to match the length of requests (prompts)
        if locality_inputs is not None:
            if len(locality_inputs) < len(prompts):
                locality_inputs = (locality_inputs * math.ceil(len(prompts) / len(locality_inputs)))[:len(prompts)]
                random.shuffle(locality_inputs)  # Shuffle to randomize the locality input order

        # Replicate portability_inputs if necessary to match the length of requests (prompts)
        if portability_inputs is not None:
            if len(portability_inputs) < len(prompts):
                portability_inputs = (portability_inputs * math.ceil(len(prompts) / len(portability_inputs)))[:len(prompts)]
                random.shuffle(portability_inputs)  # Shuffle to randomize the portability input order

        # Prepare image paths and load images
        image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in image]
        images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in image_path]
        if 'llava' in self.hparams.model_name:
            images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in images]
        else:
            images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in images]
        
        # Create requests list
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        } for prompt, target, image_ in zip(prompts, targets, images)]

        # Handle 'subject' keyword in kwargs
        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        'subject': request["prompt"].split()[-1]
                    }
                )

        # Handle rephrase prompts
        if rephrase_prompts is not None:
            for i, request in enumerate(requests):
                request.update({
                    'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if rephrase_prompts[i] and request['image'] is not None else rephrase_prompts[i],
                })
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            if 'llava' in self.hparams.model_name:
                rephrase_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            else:
                rephrase_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        # Handle locality inputs (text and vision)
        if locality_inputs is not None:
            for i, locality_input in enumerate(locality_inputs):
                request = requests[i]
                if "text" in locality_input:
                    locality_prompts = locality_input['text']['prompt']
                    locality_ground_truth = locality_input['text']['ground_truth']
                    locality_prompts = [locality_prompts] if isinstance(locality_prompts, str) else locality_prompts
                    locality_ground_truth = [locality_ground_truth] if isinstance(locality_ground_truth, str) else locality_ground_truth
                    request.update(
                        {
                            'locality_prompt': locality_prompts[0],
                            'locality_ground_truth':locality_ground_truth[0]
                        }
                    )
                # One sample has one locality and portability input, return index 0, if there are multiple locality inputs, remove[0] 
                # Vision locality
                if "vision" in locality_input:
                    vision_prompts = locality_input['vision']['prompt']
                    vision_ground_truth = locality_input['vision']['ground_truth']
                    vision_images = locality_input['vision']['image']
                    vision_prompts = [vision_prompts] if isinstance(vision_prompts, str) else vision_prompts
                    vision_ground_truth = [vision_ground_truth] if isinstance(vision_ground_truth, str) else vision_ground_truth
                    vision_images = [vision_images] if isinstance(vision_images, str) else vision_images
                    vision_images_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in vision_images]
                    vision_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in vision_images_path]
                    if 'llava' in self.hparams.model_name:
                        vision_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    else:
                        vision_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    request.update(
                        {
                            'multimodal_locality_image': vision_images[0],
                            'multimodal_locality_prompt': [self.prompt.format(item) for item in vision_prompts][0],
                            'multimodal_locality_ground_truth': vision_ground_truth[0]
                        }
                    )
        # Handle portability inputs (text and vision)
        if portability_inputs is not None:
            for i, portability_input in enumerate(portability_inputs):
                request = requests[i]
                if "text" in portability_input:
                    portability_prompts = portability_input['text']['prompt']
                    portability_ground_truth = portability_input['text']['ground_truth']
                    portability_image = portability_input['text']['image']
                    portability_prompts = [portability_prompts] if isinstance(portability_prompts, str) else portability_prompts
                    portability_ground_truth = [portability_ground_truth] if isinstance(portability_ground_truth, str) else portability_ground_truth
                    portability_image = [portability_image] if isinstance(portability_image, str) else portability_image
                    portability_image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in portability_image]
                    portability_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
                    if 'llava' in self.hparams.model_name:
                        portability_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_images]
                    else:
                        portability_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_images]
                    request.update(
                        {
                            'portability_prompt': [item for item in portability_prompts],
                            'portability_ground_truth': portability_ground_truth,
                            'portability_image': portability_images
                        }
                    )
                   

                # Vision portability
                if "vision" in portability_input:
                    vision_prompts = portability_input['vision']['prompt']
                    vision_ground_truth = portability_input['vision']['ground_truth']
                    vision_images = portability_input['vision']['image']
                    vision_prompts = [vision_prompts] if isinstance(vision_prompts, str) else vision_prompts
                    vision_ground_truth = [vision_ground_truth] if isinstance(vision_ground_truth, str) else vision_ground_truth
                    vision_images = [vision_images] if isinstance(vision_images, str) else vision_images
                    vision_images_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in vision_images]
                    vision_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in vision_images_path]
                    if 'llava' in self.hparams.model_name:
                        vision_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    else:
                        vision_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    request.update(
                        {
                            'multimodal_portability_prompt': [self.prompt.format(item) for item in vision_prompts][0],
                            'multimodal_portability_ground_truth': vision_ground_truth[0],
                            'multimodal_portability_image': vision_images[0],
                        }
                    )

        return requests
    def batch_edit_unke(self,
            prompts: List[str],
            targets: List[str],
            images: List[str],
            rephrase_prompts: Optional[List[str]] = None,
            rephrase_images: Optional[List[str]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            sequential_edit=False,
            verbose=True,
            **kwargs):
        """
        Perform batch multimodal editing.

        `prompts`: List of text prompts to edit.
        `targets`: List of expected output texts.
        `image_paths`: List of image file paths for multimodal input.
        """
        # assert len(prompts) == len(targets) == len(images), "Input lists must have the same length"

        if isinstance(self.hparams.device, str):
            self.hparams.device = str(self.model.llava_model.device).split(":")[1]
        # self.hparams.device = str(self.model.llava_model.device)
        # Prepare requests
        requests = self._prepare_requests_batch_for_unke(prompts, targets, images, rephrase_prompts, rephrase_images, locality_inputs, portability_inputs, **kwargs)
        
        assert hasattr(self.hparams, 'batch_size'), "Please specify batch_size in hparams."

        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            # Apply the editing algorithm to the batch of requests
            if self.alg_name in ['MEMIT','UnKE','AlphaEdit']:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    record_chunks,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs.get('train_ds', None) if self.alg_name == 'IKE' else None
                )
            else: 
                assert f"{self.alg_name} does not support batch edit!"
                

            exec_time = time() - start
            LOG.info(f"Batch execution took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):
                # Calculate metrics
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                    "post": self.compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                            request, self.hparams.device),
                }
                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i].update(
                    {
                        "pre":self.compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                            request, self.hparams.device)
                    }
                )

                if verbose:
                    LOG.info(f"{i} editing: {request['prompt']} -> {request['target']}")
                    pprint.pprint(chunk_metrics[i])

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        
        return all_metrics, edited_model, weights_copy

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
                    
    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }
        
        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
            
        return data
    
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in image]
        image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in image_path]
        if 'llava' in self.hparams.model_name:
            image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in image]
        else:
            image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in image]
        
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        'subject': request["prompt"].split()[-1]
                    }
                )

        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_prompts) == len(locality_ground_truth) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(multimodal_locality_prompts) == len(multimodal_locality_ground_truth) \
                == len(multimodal_locality_image) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if request['image'] is not None else rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            if 'llava' in self.hparams.model_name:
                rephrase_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            else:
                rephrase_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            
            locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) if multimodal_locality_image_ is not None else None for multimodal_locality_image_ in multimodal_locality_image]
            locality_image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in locality_image_path]
            if 'llava' in self.hparams.model_name:
                locality_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in locality_image]
            else:
                locality_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in locality_image]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': self.prompt.format(multimodal_locality_prompts[i]) if locality_image[i] is not None else multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
        
        if "text" in portability_inputs.keys():
            portability_prompts = portability_inputs['text']['prompt']
            portability_ground_truth = portability_inputs['text']['ground_truth']
            portability_image= portability_inputs['text']['image']
            if isinstance(portability_prompts, str):
                portability_prompts = [portability_prompts, ]
            if isinstance(portability_ground_truth, str):
                portability_ground_truth = [portability_ground_truth, ]
            if isinstance(portability_image, str):
                portability_image = [portability_image, ]
            assert len(portability_prompts) == len(portability_ground_truth) \
                == len(portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in portability_inputs.keys():
            multimodal_portability_prompts = portability_inputs['vision']['prompt']
            multimodal_portability_ground_truth = portability_inputs['vision']['ground_truth']
            multimodal_portability_image = portability_inputs['vision']['image']
            if isinstance(multimodal_portability_prompts, str):
                multimodal_portability_prompts = [multimodal_portability_prompts, ]
            if isinstance(multimodal_portability_ground_truth, str):
                multimodal_portability_ground_truth = [multimodal_portability_ground_truth, ]
            if isinstance(multimodal_portability_image, str):
                multimodal_portability_image = [multimodal_portability_image, ]
            assert len(multimodal_portability_prompts) == len(multimodal_portability_ground_truth) \
                == len(multimodal_portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
    

        if "text" in portability_inputs.keys():
            portability_image_path = [os.path.join(self.vis_root, portability_image_) if portability_image_ is not None else None for portability_image_ in portability_image]
            portability_image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
            if 'llava' in self.hparams.model_name:
                portability_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            else:
                portability_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'portability_prompt': self.prompt.format(portability_prompts[i]) if portability_image[i] is not None else portability_prompts[i],
                        'portability_ground_truth': portability_ground_truth[i],
                        'portability_image': portability_image[i]
                    }
                )
        
        if "vision" in portability_inputs.keys():
            portability_image_path = [os.path.join(self.vis_root, multimodal_portability_image_) if multimodal_portability_image_ is not None else None for multimodal_portability_image_ in multimodal_portability_image]
            portability_image = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
            if 'llava' in self.hparams.model_name:
                portability_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            else:
                portability_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_image]
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_portability_image': portability_image[i],
                        'multimodal_portability_prompt': self.prompt.format(multimodal_portability_prompts[i]) if portability_image[i] is not None else multimodal_portability_prompts[i],
                        'multimodal_portability_ground_truth': multimodal_portability_ground_truth[i],
                    }
                )
        return requests
    
    def _prepare_requests_batch(self,
        prompts: Union[str, List[str]],
        targets: Union[str, List[str]],
        image: Union[str, List[str]],
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        rephrase_image: Optional[Union[str, List[str]]] = None,
        locality_inputs: Optional[List[Dict]] = None,
        portability_inputs: Optional[List[Dict]] = None,
        **kwargs):
        # Ensure that inputs are lists if they are not already
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(image, str):
            image = [image]
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts]
        if isinstance(rephrase_image, str):
            rephrase_image = [rephrase_image]
        if isinstance(locality_inputs, dict):
            locality_inputs = [locality_inputs]
        if isinstance(portability_inputs, dict):
            portability_inputs = [portability_inputs]

        # Ensure that all lists have the same length
        assert len(prompts) == len(targets) == len(image), "Prompts, targets, and images must have the same length"

        # Replicate locality_inputs if necessary to match the length of requests (prompts)
        if locality_inputs is not None:
            if len(locality_inputs) < len(prompts):
                locality_inputs = (locality_inputs * math.ceil(len(prompts) / len(locality_inputs)))[:len(prompts)]
                random.shuffle(locality_inputs)  # Shuffle to randomize the locality input order

        # Replicate portability_inputs if necessary to match the length of requests (prompts)
        if portability_inputs is not None:
            if len(portability_inputs) < len(prompts):
                portability_inputs = (portability_inputs * math.ceil(len(prompts) / len(portability_inputs)))[:len(prompts)]
                random.shuffle(portability_inputs)  # Shuffle to randomize the portability input order

        # Prepare image paths and load images
        image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in image]
        images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in image_path]
        if 'llava' in self.hparams.model_name:
            images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in images]
        else:
            images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in images]
        
        # Create requests list
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        } for prompt, target, image_ in zip(prompts, targets, images)]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        'subject': request["prompt"].split()[-1]
                    }
                )

        # Handle rephrase prompts
        if rephrase_prompts is not None:
            for i, request in enumerate(requests):
                request.update({
                    'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if rephrase_prompts[i] and request['image'] is not None else rephrase_prompts[i],
                })
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            if 'llava' in self.hparams.model_name:
                rephrase_image = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            else:
                rephrase_image = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        # Handle locality inputs (text and vision)
        if locality_inputs is not None:
            for i, locality_input in enumerate(locality_inputs):
                request = requests[i]
                if "text" in locality_input:
                    locality_prompts = locality_input['text']['prompt']
                    locality_ground_truth = locality_input['text']['ground_truth']
                    locality_prompts = [locality_prompts] if isinstance(locality_prompts, str) else locality_prompts
                    locality_ground_truth = [locality_ground_truth] if isinstance(locality_ground_truth, str) else locality_ground_truth
                    request.update(
                        {
                            'locality_prompt': locality_prompts[0],
                            'locality_ground_truth':locality_ground_truth[0]
                        }
                    )
                # One sample has one locality and portability input, return index 0, if there are multiple locality inputs, remove[0] 
                # Vision locality
                if "vision" in locality_input:
                    vision_prompts = locality_input['vision']['prompt']
                    vision_ground_truth = locality_input['vision']['ground_truth']
                    vision_images = locality_input['vision']['image']
                    vision_prompts = [vision_prompts] if isinstance(vision_prompts, str) else vision_prompts
                    vision_ground_truth = [vision_ground_truth] if isinstance(vision_ground_truth, str) else vision_ground_truth
                    vision_images = [vision_images] if isinstance(vision_images, str) else vision_images
                    vision_images_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in vision_images]
                    vision_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in vision_images_path]
                    if 'llava' in self.hparams.model_name:
                        vision_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    else:
                        vision_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    request.update(
                        {
                            'multimodal_locality_image': vision_images[0],
                            'multimodal_locality_prompt': [self.prompt.format(item) for item in vision_prompts][0],
                            'multimodal_locality_ground_truth': vision_ground_truth[0]
                        }
                    )
        # Handle portability inputs (text and vision)
        if portability_inputs is not None:
            for i, portability_input in enumerate(portability_inputs):
                request = requests[i]
                if "text" in portability_input:
                    portability_prompts = portability_input['text']['prompt']
                    portability_ground_truth = portability_input['text']['ground_truth']
                    portability_image = portability_input['text']['image']
                    portability_prompts = [portability_prompts] if isinstance(portability_prompts, str) else portability_prompts
                    portability_ground_truth = [portability_ground_truth] if isinstance(portability_ground_truth, str) else portability_ground_truth
                    portability_image = [portability_image] if isinstance(portability_image, str) else portability_image
                    portability_image_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in portability_image]
                    portability_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in portability_image_path]
                    if 'llava' in self.hparams.model_name:
                        portability_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_images]
                    else:
                        portability_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in portability_images]
                    request.update(
                        {
                            # 'portability_prompt': [self.prompt.format(item) for item in portability_prompts][0],
                            'portability_prompt': [item for item in portability_prompts][0],
                            'portability_ground_truth': portability_ground_truth[0],
                            'portability_image': portability_images[0]
                        }
                    )
                   

                # Vision portability
                if "vision" in portability_input:
                    vision_prompts = portability_input['vision']['prompt']
                    vision_ground_truth = portability_input['vision']['ground_truth']
                    vision_images = portability_input['vision']['image']
                    vision_prompts = [vision_prompts] if isinstance(vision_prompts, str) else vision_prompts
                    vision_ground_truth = [vision_ground_truth] if isinstance(vision_ground_truth, str) else vision_ground_truth
                    vision_images = [vision_images] if isinstance(vision_images, str) else vision_images
                    vision_images_path = [os.path.join(self.vis_root, image_) if image_ is not None else None for image_ in vision_images]
                    vision_images = [Image.open(ip).convert("RGB") if ip is not None else None for ip in vision_images_path]
                    if 'llava' in self.hparams.model_name:
                        vision_images = [self.vis_tok.preprocess(i, return_tensors='pt')['pixel_values'].half().to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    else:
                        vision_images = [self.vis_tok(i).to(f"cuda:{self.hparams.device}") if i is not None else None for i in vision_images]
                    request.update(
                        {
                            'multimodal_portability_prompt': [self.prompt.format(item) for item in vision_prompts][0],
                            'multimodal_portability_ground_truth': vision_ground_truth[0],
                            'multimodal_portability_image': vision_images[0],
                        }
                    )

        return requests

    def _prepare_requests_dataset(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
            
        requests = [{
            'prompt': self.prompt.format(prompt) if image_ is not None else prompt,
            'target': target,
            'image': image_,
            'prompt_template': self.prompt_template,
            'image_toks': self.image_toks,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        else:
            for request in requests:
                request.update(
                    {
                        'subject': request["prompt"].split()[-1]
                    }
                )

        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_prompts) == len(locality_ground_truth) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(multimodal_locality_prompts) == len(multimodal_locality_ground_truth) \
                == len(multimodal_locality_image) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': self.prompt.format(rephrase_prompts[i]) if request['image'] is not None else rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': multimodal_locality_image[i],
                        'multimodal_locality_prompt': self.prompt.format(multimodal_locality_prompts[i]) if multimodal_locality_image[i] is not None else multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
        
        if "text" in portability_inputs.keys():
            portability_prompts = portability_inputs['text']['prompt']
            portability_ground_truth = portability_inputs['text']['ground_truth']
            portability_image= portability_inputs['text']['image']
            if isinstance(portability_prompts, str):
                portability_prompts = [portability_prompts, ]
            if isinstance(portability_ground_truth, str):
                portability_ground_truth = [portability_ground_truth, ]
            if isinstance(portability_image, str):
                portability_image = [portability_image, ]
            assert len(portability_prompts) == len(portability_ground_truth) \
                == len(portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in portability_inputs.keys():
            multimodal_portability_prompts = portability_inputs['vision']['prompt']
            multimodal_portability_ground_truth = portability_inputs['vision']['ground_truth']
            multimodal_portability_image = portability_inputs['vision']['image']
            if isinstance(multimodal_portability_prompts, str):
                multimodal_portability_prompts = [multimodal_portability_prompts, ]
            if isinstance(multimodal_portability_ground_truth, str):
                multimodal_portability_ground_truth = [multimodal_portability_ground_truth, ]
            if isinstance(multimodal_portability_image, str):
                multimodal_portability_image = [multimodal_portability_image, ]
            assert len(multimodal_portability_prompts) == len(multimodal_portability_ground_truth) \
                == len(multimodal_portability_image) == len(requests) or print('One Edit instance needs one locality input.....')
    

        if "text" in portability_inputs.keys():
            for i, request in enumerate(requests):
                request.update(
                    {
                        'portability_prompt': self.prompt.format(portability_prompts[i]) if portability_image[i] is not None else portability_prompts[i],
                        'portability_ground_truth': portability_ground_truth[i],
                        'portability_image': portability_image[i]
                    }
                )
        
        if "vision" in portability_inputs.keys():
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_portability_image': multimodal_portability_image[i],
                        'multimodal_portability_prompt': self.prompt.format(multimodal_portability_prompts[i]) if multimodal_portability_image[i] is not None else multimodal_portability_prompts[i],
                        'multimodal_portability_ground_truth': multimodal_portability_ground_truth[i],
                    }
                )
        return requests