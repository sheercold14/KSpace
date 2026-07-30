from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .loranull_hparams import LoRANULLMultimodalHyperParams
from ...trainer.losses import masked_log_probs

from tqdm import tqdm

# Multimodal dataset for Corda 
from ...dataset import VQADataset_Simple
from torch.utils.data import DataLoader
import gc
from .adapterlib.decomposition import build_model2
from .adapterlib.datautils import get_calib_data
from .adapterlib.act_aware_utils import calib_cov_distribution

def _logits(x):
    return x if not hasattr(x, "logits") else x.logits


def apply_loranull_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRANULLMultimodalHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model) 
        if hparams.cpu_copy:
            model = model.to("cuda")

    if hparams.cpu_copy:
        model = model.to("cuda") 

    edited_model = execute_lora(model, tok, requests, hparams, keep_original_weight)
    if hasattr(model, "llava_model") or hasattr(model, "qwen_model") or hasattr(model, "phi_model"):
        return model, weights_copy
    else:
        return edited_model, weights_copy


def execute_lora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRANULLMultimodalHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the Lora update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # for sub_model_name in ['llava_model', '']:
    #     sub_model = getattr(model, sub_model_name)
    #     if sub_model and hasattr(sub_model, 'config'):
    #         llava_model = sub_model
    #         break
    # model.config.use_cache = False
    # model.supports_gradient_checkpointing = True  #
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    if hasattr(hparams, 'exclude_modules'):
        if hparams.model_name in ['qwen2.5_vl']:
            exclude_modules = [
                f"visual.blocks.{layer}.mlp.{module}"
                for layer in hparams.layers
                for module in hparams.target_modules
            ]
        elif hparams.model_name in ['phi3_vl', 'phi4_vl']:
            exclude_modules = [
                f"model.embed_tokens_extend.image_embed.img_processor.encoder.layers.{layer}.self_attn.{module}"
                for layer in hparams.layers
                for module in hparams.target_modules
            ]
        elif hparams.model_name in ['llava']:
            exclude_modules = [
                f"model.llava_model.model.vision_tower.vision_tower.vision_model.encoder.layers.{layer}.self_attn.{module}"
                for layer in hparams.layers
                for module in hparams.target_modules
            ]

            exclude_modules = [
                f"vision_tower.vision_tower.vision_model.encoder.layers.{layer}.self_attn.{module}"
                for layer in hparams.layers
                for module in hparams.target_modules
            ]
        else:
            assert False, f"Unsupported model {hparams.model_name} for LoRA"
        # exclude_modules = hparams.exclude_modules
    else:
        exclude_modules = ["vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj", "vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.v_proj"]

    

    if hasattr(model, "llava_model"):
        sub_model = model.llava_model
    elif hasattr(model, "qwen_model"):
        sub_model = model.qwen_model
    elif hasattr(model, "phi_model"):
        sub_model = model.phi_model
    else:
        sub_model = model
    sub_model.config.use_cache = False
    sub_model.supports_gradient_checkpointing = True  #
    sub_model.gradient_checkpointing_enable()
    sub_model.enable_input_require_grads()
    if hparams.Null_mode:
        for n, p in sub_model.named_parameters():
            ## freeze BLinaer
            # and "BLinear" not in n 
            if "ALinear" not in n and p.requires_grad:
                p.requires_grad = False
            if ("PALinear" in n or "PBLinear"in n )and p.requires_grad:
                p.requires_grad = False
    elif hparams.lora_type == "lora":
        Config = LoraConfig
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules,
            total_step=hparams.num_steps
        )
    elif hparams.lora_type == "corda":
        # sampled_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]", ignore_verifications=True)
        # sampled_dataset = load_dataset("wikipedia", '20200501.en', split="train[:256]")
        if hparams.model_name == 'llava':
            from ...trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
            prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
            template = requests[0]["prompt_template"]
        if prompt:
            ds = get_VQA_ds(hparams,prompt,template,size=30) 
        else:
            assert "No prompt is defined for multimodal text inputs"
        dataloader = DataLoader(
            ds,
            batch_size=1,  # You can change this depending on your batch size requirement
            shuffle=False,  # Shuffle the data to ensure randomness in training
            collate_fn=ds.collate_fn  # Pass the custom collate_fn defined in the dataset
        )
        # dataset = load_dataset("imdb", split="train[:256]")
        # def run_model():
            # for batch in tqdm(sampled_dataset):
            #     samples = [
            #         {
            #             "text_input": [batch["text"]],
            #             "image": None,
            #         }
            #     ][0]
            #     with torch.no_grad():
            #         model(samples)
        def run_model():
            for batch in tqdm(dataloader):
                samples = {
                        "text_input": batch["text_input"],
                        "image": batch["image"],
                    }
            
                with torch.no_grad():
                    model(samples)
        corda_config = CordaConfig(
            corda_method="kpm",
            covariance_file=getattr(
                hparams, "corda_covariance_file", "outputs/cache/corda/covariance.pt"
            ),
            cache_file=getattr(
                hparams, "corda_cache_file", "outputs/cache/corda/cache.pt"
            ),
        )
        peft_config = LoraConfig(
            init_lora_weights="corda",
            corda_config=corda_config,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules,
            exclude_modules=exclude_modules
        )
        preprocess_corda(sub_model, lora_config=peft_config, run_model=run_model)
    else:
        raise NotImplementedError

    if hparams.Null_mode:
        peft_model = sub_model
    elif not keep_original_weight and hasattr(model, 'peft_config'):
        peft_model = sub_model
    else:
        peft_model = get_peft_model(sub_model, peft_config).to(torch.bfloat16)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    if hasattr(peft_model, 'print_trainable_parameters'):
        peft_model.print_trainable_parameters()
    requests = deepcopy(requests)
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
        print(
            f"Executing LoRA algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    device = torch.device(f'cuda:{hparams.device}')
    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    prompt_template = "{}" if requests[0]["prompt_template"] is None else requests[0]["prompt_template"]
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    if "image" in requests[0]:
        images = [r["image"] for r in requests]
    # if torch.__version__ >= "2" and sys.platform != "win32":
    # model = torch.compile(model)
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt, img in zip(
                chunks(texts, hparams.batch_size), 
                chunks(targets, hparams.batch_size),
                chunks(images, hparams.batch_size)
        ):
            mask_token = -100
            opt.zero_grad()
            if 't5' in hparams.model_name.lower():
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                bs = inputs["input_ids"].shape[0]
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                    device
                )
                inputs['labels'] = target_ids
                logits = peft_model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)
                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:
                # src_trg_inputs = tok(txt + tgt, return_tensors="pt", padding=True).to(device)
                # bs = src_trg_inputs["input_ids"].shape[0]
                # targ = deepcopy(src_trg_inputs['input_ids'])
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # targ = targ[:, 1:]
                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
                # eos_token = tok.decode(tok.eos_token_id)
                if img:
                    if "qwen2.5_vl" in hparams.model_name or "phi3_vl" in hparams.model_name or "phi4_vl" in hparams.model_name:
                        full_prompt = [p for p in txt]
                        answer = [l for l in tgt]
                        samples = {
                            "noise": True,
                            "text_input": full_prompt,
                            "image": img,
                            "train": True,
                            "answer": answer
                        }
                    else:    
                        full_prompt = [f"{prompt_template.format(p)} {l}" for p, l in zip(txt, tgt)]
                        samples = {
                            "noise": True,
                            "text_input": full_prompt,
                            "image": img,
                            "train": True,
                        }
                    # pred = model(samples, output_attentions=False)
                    if isinstance(tgt, list):
                        tgt = tgt[0]
                    if "phi4_vl" in hparams.model_name or "phi3_vl" in hparams.model_name:
                        loss = model(samples, output_attentions=False,freeze_partial_params=True).loss
                    elif "qwen2.5_vl" in hparams.model_name:
                        loss = model(samples, output_attentions=False).loss
                    else:
                        labels = tok.encode(tgt, add_special_tokens=False,return_tensors="pt").to(device)
                        logits = _logits(model(samples))
                        loss = masked_log_probs(hparams, logits, labels, shift=True)["nll"]
                else:
                    full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                    prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
                    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
                    bs = tokens["input_ids"].shape[0]
                    tokens["labels"] = tokens["input_ids"].clone()
                    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
                    for i in range(len(txt)):
                        tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
                    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                    tokens = tokens.to(device)
                    pred = peft_model(**tokens)
                    loss = pred.loss

            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=len(full_prompt))

            # if loss.item() >= 1e-3:

            loss.backward()
            opt.step()

        print(f"Total loss {loss_meter.avg}")

        # if loss_meter.avg < 1e-3:
        #     break
    return peft_model


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def get_VQA_ds(hparams, prompt, template, size=None):
    annotation_path = hparams.train_annotation_path
    image_root = hparams.coco_image
    raw_ds = VQADataset_Simple(size=size, prompt=prompt,template=template,annotation_file=annotation_path,image_root=image_root,image_size=336)
    return raw_ds
