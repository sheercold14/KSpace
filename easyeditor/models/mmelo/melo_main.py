from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib

from .melo_hparams import MELOMultimodalHyperParams
from .algs.lora import LORA

from .multimodal_trainer import vqa_trainer, caption_trainer
from ...evaluate import prepare_multimodal_edit_batch

from .database.router import Router

class MMelo:
    def __init__(self,       
            model: AutoModelForCausalLM,
            tok: AutoTokenizer,
            hparams: MELOMultimodalHyperParams,
            copy=False,
            **kwargs: Any) -> None:
        self.hparams = hparams
        if copy:
            model = deepcopy(model)
        self.tok = tok
        self.alg = LORA(model, hparams)
        self.alg.to(hparams.device)
        self.router = Router(self.hparams)

    def run(
        self,        
        requests: List[Dict],
        **kwargs: Any
    )-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
        if "idx" in kwargs:
            idx = kwargs.get("idx", 0)
        else:
            raise ValueError("idx not found in kwargs")
        weights_copy = {}
        edit_inner = prepare_multimodal_edit_batch(self.hparams, self.tok, batch=requests, prompt_template=requests[0]['prompt_template'])
        if self.hparams.task == "caption":
            trainer = caption_trainer(self.router, self.alg, dict_to({"edit_inner": edit_inner}, self.hparams.device), idx)
        elif self.hparams.task == "vqa":
            trainer = vqa_trainer(self.router, self.alg, dict_to({"edit_inner": edit_inner}, self.hparams.device), idx)
            
        torch.cuda.empty_cache()
        trainer.run_edit()
        
        return self.alg, self.router, weights_copy


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict
