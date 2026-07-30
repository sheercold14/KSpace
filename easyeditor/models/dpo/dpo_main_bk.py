from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dpo_hparams import DPOMultimodalHyperParams, DPOHyperParams
from accelerate import Accelerator

def apply_dpo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    """
    weights_copy = {}
    if copy:
        # If you need to copy the model, handle it here
        pass  # Avoid deep copying to save memory

    device = torch.device(f'cuda:{hparams.device}')
    print(f"Using device: {device}")

    # Configure LoRA
    Config = LoraConfig

    peft_config = Config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules
    )

    target_peft_model = model.llava_model if hasattr(model, 'llava_model')  else model
    # Add LoRA modules to the model
    peft_model = get_peft_model(target_peft_model, peft_config)
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    peft_model.to(device)

    # Manually set only LoRA parameters to be trainable
    for name, param in peft_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    peft_model.to(device)

    # Execute the DPO algorithm
    edited_model = execute_dpo(model, peft_model, tok, requests, hparams)
    
    model.llava_model = edited_model
    edited_model = model

    return edited_model, weights_copy


def execute_dpo(
        model: AutoModelForCausalLM,
        peft_model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the DPO algorithm for the specified updates.
    """
    peft_model.train()
    device = next(peft_model.parameters()).device

    # Define the optimizer
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    
    accelerator = Accelerator()

    peft_model, optimizer = accelerator.prepare(peft_model, opt)

    loss_meter = AverageMeter()

    # Prepare data
    texts = [r["prompt_template"].format(r["prompt"]) if "prompt_template" in r else r["prompt"] for r in requests]
    targets_pos = [r["target"] for r in requests]  # Positive samples
    targets_neg = [r["targets_neg"] for r in requests]  # Negative samples
    
    if "image" in requests[0]:
        images = [r["image"] for r in requests]  # Images
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt_batch, tgt_pos_batch, tgt_neg_batch, image_batch in zip(
                chunks(texts, hparams.batch_size),
                chunks(targets_pos, hparams.batch_size),
                chunks(targets_neg, hparams.batch_size),
                chunks(images, hparams.batch_size) if "image" in requests[0] else None,
        ):
            mask_token = -100
            opt.zero_grad()

            if image_batch:
                full_prompt_pos = [f"{p} {l}" for p, l in zip(txt_batch, tgt_pos_batch)]
                samples_pos = {
                    "noise": True,
                    "text_input": full_prompt_pos,
                    "image": image_batch,
                    "train": True
                }
                
                full_prompt_neg = [f"{p} {l}" for p, l in zip(txt_batch, tgt_neg_batch)]
                samples_neg = {
                    "noise": True,
                    "text_input": full_prompt_neg,
                    "image": image_batch,
                    "train": True
                }
                # Compute outputs with LoRA modules (current model)
                outputs_pos = model(samples_pos, output_attentions=False)
                outputs_neg = model(samples_neg, output_attentions=False)
            else:
                # Build inputs for positive samples
                full_prompt_pos = [f"{p} {l}" for p, l in zip(txt_batch, tgt_pos_batch)]
                tokens_pos = tok(full_prompt_pos, return_tensors="pt", padding=True, truncation=True)
                tokens_pos["labels"] = tokens_pos["input_ids"].clone()
                tokens_pos["labels"][tokens_pos["input_ids"] == tok.pad_token_id] = mask_token
                tokens_pos = tokens_pos.to(device)

                # Build inputs for negative samples
                full_prompt_neg = [f"{p} {l}" for p, l in zip(txt_batch, tgt_neg_batch)]
                tokens_neg = tok(full_prompt_neg, return_tensors="pt", padding=True, truncation=True)
                tokens_neg["labels"] = tokens_neg["input_ids"].clone()
                tokens_neg["labels"][tokens_neg["input_ids"] == tok.pad_token_id] = mask_token
                tokens_neg = tokens_neg.to(device)

                # Compute outputs with LoRA modules (current model)
                outputs_pos = peft_model(**tokens_pos)
                outputs_neg = peft_model(**tokens_neg)

            # Compute outputs for the reference model (disable LoRA modules)
            peft_model.eval()  # Switch to evaluation mode
            peft_model.disable_adapter_layers()  # Disable LoRA layers

            with torch.no_grad():
                if image_batch:
                    ref_outputs_pos = model(samples_pos, output_attentions=False)
                    ref_outputs_neg = model(samples_neg, output_attentions=False)
                else:
                    ref_outputs_pos = peft_model(**tokens_pos)
                    ref_outputs_neg = peft_model(**tokens_neg)

            peft_model.train()  # Switch back to training mode
            peft_model.enable_adapter_layers()  # Enable LoRA layers

            # Compute losses
            lora_loss = outputs_pos.loss
            beta = hparams.beta

            ref_log_probs_pos = ref_outputs_pos.logits.log_softmax(-1)
            ref_log_probs_neg = ref_outputs_neg.logits.log_softmax(-1)

            log_probs_pos = outputs_pos.logits.log_softmax(-1)
            log_probs_neg = outputs_neg.logits.log_softmax(-1)

            dpo_advantage = beta * (
                (log_probs_pos - ref_log_probs_pos).sum(-1) -
                (log_probs_neg - ref_log_probs_neg).sum(-1)
            )
            # dpo_loss = -torch.mean(torch.log(torch.sigmoid(dpo_advantage)))
            dpo_loss = -torch.nn.functional.logsigmoid(dpo_advantage).mean()

            # Total loss
            loss = hparams.alpha * lora_loss + (1 - hparams.alpha) * dpo_loss

            loss.backward()
            opt.step()

            bs = len(txt_batch)
            loss_meter.update(loss.item(), n=bs)

        print(f"Total loss {loss_meter.avg}")

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
    for i in range(0, len(arr), n):
        yield arr[i:i + n]
