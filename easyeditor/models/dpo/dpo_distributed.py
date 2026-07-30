import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from torch.optim import Adam
# from accelerate import Accelerator # Removed Accelerate imports
from .dpo_hparams import DPOMultimodalHyperParams, DPOHyperParams
# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os # Needed for environment variables

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assume these are defined elsewhere
# from .dpo_hparams import DPOMultimodalHyperParams, DPOHyperParams
# Placeholder for HyperParams class
class DPOHyperParams:
    def __init__(self):
        self.rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.layers = []
        self.target_modules = ["q_proj", "v_proj"] # Example, adjust as needed
        self.batch_size = 2 # Per GPU batch size
        self.num_steps = 3 # Epochs
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.beta = 0.1 # DPO beta
        self.alpha = 0.5 # Loss weighting alpha (0.5 means equal weight)
        # device parameter is removed, determined by local_rank now

class DPODataset(Dataset):
    """
    将原先的 `requests` 列表封装成可迭代的 Dataset。
    每个 item 包含: prompt(文本), target(正样本), targets_neg(负样本), image(可选)
    """
    def __init__(self, requests):
        super().__init__()
        self.requests = requests

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        r = self.requests[idx]
        # 如果有 prompt_template 就先 format，否则用 prompt
        text = r["prompt_template"].format(r["prompt"]) if "prompt_template" in r else r["prompt"]
        pos = r["target"]
        neg = r["targets_neg"]
        # 若是多模态，就可能有 image
        img = r.get("image", None) # Assuming image is already a Tensor or PIL Image
        return {
            "text": text,
            "pos": pos,
            "neg": neg,
            "image": img
        }

def dpo_collate_fn(batch):
    """
    将一批 item (dict) 合并成一个 batch dict。
    Images are kept as a list. If they are tensors, they should be handled later.
    """
    texts = [x["text"] for x in batch]
    pos_targets = [x["pos"] for x in batch]
    neg_targets = [x["neg"] for x in batch]
    # Handle images carefully. If they are PIL Images, keep as list.
    # If they need to be tensors, stacking might require pre-processing.
    # For now, just keep them as a list. The model's forward should handle it.
    images = [x["image"] for x in batch]
    return {
        "texts": texts,
        "pos_targets": pos_targets,
        "neg_targets": neg_targets,
        "images": images,
    }

class AverageMeter:
    """Computes and stores the average and current value."""
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

# --- Distributed Setup ---
def setup_distributed(backend='nccl'):
    """Initializes the distributed environment."""
    if dist.is_initialized():
        return
    # Initializes the distributed backend
    # RANK, WORLD_SIZE, LOCAL_RANK are set by the launch script (e.g., torchrun)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"Rank {rank}/{world_size}, Local Rank {local_rank}: Initializing process group...")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size}, Local Rank {local_rank}: Process group initialized. Using CUDA device {local_rank}.")

def cleanup_distributed():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

# --- Main DPO Application Function ---
def apply_dpo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        copy=False, # copy/return_orig_weights logic remains placeholder
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    使用 DDP 对给定模型执行DPO。
    1) 给 model.llava_model (或 model 本身) 添加 LoRA
    2) 创建 Dataset + DistributedSampler + DataLoader
    3) 调用 execute_dpo 进行多卡训练
    """

    # --- DDP Setup ---
    setup_distributed()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"Starting DPO on {world_size} GPUs.")
        print(f"Using device: {device} (assigned to rank {rank}, local_rank {local_rank})")

    # Optional weight copy logic (unchanged)
    weights_copy = {}
    if copy:
        pass # Placeholder

    # Move the base model to the assigned GPU BEFORE applying PEFT or DDP
    # Note: If model is very large, consider meta device init + load_checkpoint
    model.to(device)

    # ========== 1. 准备LoRA ==========
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if hasattr(hparams, 'layers') and len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules
    )

    target_model_part = model.llava_model if hasattr(model, "llava_model") else model
    peft_model = get_peft_model(target_model_part, peft_config)
    
    # Enable grad checkpointing *before* DDP wrap
    if hasattr(peft_model, "gradient_checkpointing_enable"):
       peft_model.gradient_checkpointing_enable()
    if hasattr(peft_model, "enable_input_require_grads"):
       peft_model.enable_input_require_grads() # For gradient checkpointing with PEFT

    # Set requires_grad before DDP wrap
    # We only train LoRA parameters
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Move the PEFT model itself to the correct device (already done if target_model_part was model)
    # If target_model_part was model.llava_model, peft_model might still be on CPU, move it
    peft_model.to(device) 

    # ========== 1.5 Wrap model with DDP ==========
    # find_unused_parameters=True might be needed if not all parameters
    # wrapped by DDP have requires_grad=True (like frozen base model parts)
    # Or if gradient checkpointing hides intermediate parameter usage.
    # Start with False for efficiency, change if needed. Let's set True proactively.
    ddp_peft_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    if rank == 0:
       ddp_peft_model.module.print_trainable_parameters() # Print from underlying model on rank 0

    # ========== 2. 构建Dataset + DataLoader ==========
    dataset = DPODataset(requests)
    # DistributedSampler handles shuffling and data distribution
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # shuffle=False in DataLoader because sampler handles it
    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size, # This is now per-GPU batch size
        sampler=sampler,
        collate_fn=dpo_collate_fn,
        num_workers=4, # Example, adjust based on system
        pin_memory=True, # Helps speed up CPU -> GPU transfers
    )

    # ========== 3. 进入DPO训练循环 ==========
    # Pass the DDP-wrapped model to the execution function
    # Also pass the original base model (needed for reference passes potentially)
    # and the device for tensor placement inside the loop
    trained_peft_model_part = execute_dpo(
        base_model=model, # Pass the original model (on device)
        ddp_peft_model=ddp_peft_model, # Pass the DDP wrapped model
        tok=tok,
        dataloader=dataloader,
        hparams=hparams,
        device=device, # Pass the device for this rank
        rank=rank,     # Pass rank for logging
        world_size=world_size # Pass world size for loss sync
    )

    # Sync processes before proceeding (optional but good practice)
    dist.barrier()

    # Replace the trained part back into the original model structure (on rank 0 potentially?)
    # The trained weights are synchronized across all GPUs by DDP.
    # We can get the final state from the module attribute of the DDP wrapper.
    final_peft_model_part = ddp_peft_model.module

    # Make sure the final model state is consistent across devices if needed later
    # Typically, saving happens only on rank 0 after training.
    if hasattr(model, "llava_model"):
        model.llava_model = final_peft_model_part
    else:
        model = final_peft_model_part # This might be tricky if original 'model' was just the LM part

    # --- DDP Cleanup ---
    cleanup_distributed()
    if rank == 0:
      print("DPO Training finished and distributed processes cleaned up.")

    # Return the modified base model (which contains the trained PEFT part)
    # Note: This model exists on all ranks but only rank 0 might save it.
    return model, weights_copy

# --- DPO Execution Loop (handles DDP) ---
def execute_dpo(
        base_model: AutoModelForCausalLM, # Original model (on device)
        ddp_peft_model: DDP,         # DDP wrapped PEFT model
        tok: AutoTokenizer,
        dataloader: DataLoader,
        hparams: DPOHyperParams,
        device: torch.device,        # Device for this process
        rank: int,                   # Rank of this process
        world_size: int,             # Total number of processes
        **kwargs: Any,
) -> torch.nn.Module:
    """
    DDP training loop.
    """
    # Get the underlying model from DDP wrapper when needed (e.g., for saving, non-DDP methods)
    peft_model = ddp_peft_model.module

    # Set model to train mode (affects dropout, batchnorm etc.)
    # ddp_peft_model.train() # DDP forward takes care of setting train/eval state? Check docs. Let's be explicit.
    peft_model.train() # Set the underlying model to train

    # Optimizer operates on parameters of the model *before* DDP wrapping
    optimizer = Adam(peft_model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

    loss_meter = AverageMeter()
    mask_token = -100 # Label mask token ID

    for epoch in range(hparams.num_steps):
        # Set epoch for the sampler (important for shuffling across epochs)
        dataloader.sampler.set_epoch(epoch)

        if rank == 0:
            print("=" * 30)
            print(f"Epoch: {epoch}")
            print("=" * 30)
        loss_meter.reset()

        for step, batch in enumerate(dataloader):
            txt_batch = batch["texts"]
            tgt_pos_batch = batch["pos_targets"]
            tgt_neg_batch = batch["neg_targets"]
            img_batch = batch["images"] # List of images (PIL or Tensors)

            # Determine if multimodal based on content
            is_multimodal = any(x is not None for x in img_batch)

            # --- Prepare inputs ---
            # Combine prompts and targets
            full_prompt_pos = [f"{p} {l}" for p, l in zip(txt_batch, tgt_pos_batch)]
            full_prompt_neg = [f"{p} {l}" for p, l in zip(txt_batch, tgt_neg_batch)]

            # --- Forward Pass (Policy Model - with LoRA) ---
            optimizer.zero_grad()

            # Use ddp_peft_model for the forward pass with gradients
            ddp_peft_model.train() # Ensure train mode

            if is_multimodal:
                 # Assume base_model's forward handles list of images and texts
                 # And PEFT model is part of base_model (e.g., llava_model)
                 # The base_model structure needs modification for DPO if PEFT is separate
                 # Let's assume peft_model *is* the language model part here for simplicity
                 # This part needs careful adjustment based on actual model architecture
                 # If peft_model wraps only the LM, how does image interact?
                 # ASSUMPTION: base_model handles image+text, and ddp_peft_model is the LM part
                 # This structure might need rethinking based on LLaVa/IDEFICS integration
                 # For now, let's focus on the text-only path logic first, as multimodal adds complexity
                 # regarding how PEFT interacts with vision tower & connectors.

                 # Simplified placeholder - this needs proper multimodal handling logic
                 # that integrates the DDP-wrapped LM part correctly.
                 # This likely involves modifying base_model's forward or calling components explicitly.

                 # If base_model includes vision+connector+LM(ddp_peft_model)
                 # samples_pos = {"text_input": full_prompt_pos, "image": img_batch, "train": True} # Adapt dict keys as needed
                 # outputs_pos = base_model(samples_pos, output_attentions=False) # This base_model must use ddp_peft_model internally
                 # samples_neg = {"text_input": full_prompt_neg, "image": img_batch, "train": True}
                 # outputs_neg = base_model(samples_neg, output_attentions=False)

                 # --- Text-only path (more straightforward) ---
                 if rank == 0 and step == 0 and epoch == 0: # Warn only once
                     print("Warning: Multimodal forward pass in DDP needs careful verification based on model architecture.")
                 # For now, using text-only logic even if images present, requires adaptation
                 tokens_pos = tok(full_prompt_pos, return_tensors="pt", padding=True, truncation=True).to(device)
                 tokens_pos["labels"] = tokens_pos["input_ids"].clone()
                 tokens_pos["labels"][tokens_pos["input_ids"] == tok.pad_token_id] = mask_token
                 outputs_pos = ddp_peft_model(**tokens_pos) # Use DDP model

                 tokens_neg = tok(full_prompt_neg, return_tensors="pt", padding=True, truncation=True).to(device)
                 tokens_neg["labels"] = tokens_neg["input_ids"].clone()
                 tokens_neg["labels"][tokens_neg["input_ids"] == tok.pad_token_id] = mask_token
                 outputs_neg = ddp_peft_model(**tokens_neg) # Use DDP model

            else: # Text-only batch
                tokens_pos = tok(full_prompt_pos, return_tensors="pt", padding=True, truncation=True).to(device)
                # Create labels: mask padding tokens, rest are predicted
                labels_pos = tokens_pos["input_ids"].clone()
                labels_pos[tokens_pos["input_ids"] == tok.pad_token_id] = mask_token
                outputs_pos = ddp_peft_model(input_ids=tokens_pos.input_ids, attention_mask=tokens_pos.attention_mask, labels=labels_pos)

                tokens_neg = tok(full_prompt_neg, return_tensors="pt", padding=True, truncation=True).to(device)
                labels_neg = tokens_neg["input_ids"].clone()
                labels_neg[tokens_neg["input_ids"] == tok.pad_token_id] = mask_token
                outputs_neg = ddp_peft_model(input_ids=tokens_neg.input_ids, attention_mask=tokens_neg.attention_mask, labels=labels_neg)


            # --- Forward Pass (Reference Model - frozen, no LoRA) ---
            # Use the underlying model (peft_model) but disable adapters
            peft_model.eval() # Set dropout etc to eval
            peft_model.disable_adapter_layers()
            with torch.no_grad():
                if is_multimodal:
                    # ref_samples_pos = {"text_input": full_prompt_pos, "image": img_batch, "train": False}
                    # ref_outputs_pos = base_model(ref_samples_pos, output_attentions=False) # Base model must use peft_model internally now (adapter disabled)
                    # ref_samples_neg = {"text_input": full_prompt_neg, "image": img_batch, "train": False}
                    # ref_outputs_neg = base_model(ref_samples_neg, output_attentions=False)
                    # --- Text-only path (as fallback) ---
                    ref_outputs_pos = peft_model(**tokens_pos) # Use unwrapped model directly
                    ref_outputs_neg = peft_model(**tokens_neg) # Use unwrapped model directly
                else:
                    # Pass inputs already on device
                    ref_outputs_pos = peft_model(input_ids=tokens_pos.input_ids, attention_mask=tokens_pos.attention_mask, labels=labels_pos)
                    ref_outputs_neg = peft_model(input_ids=tokens_neg.input_ids, attention_mask=tokens_neg.attention_mask, labels=labels_neg)

            peft_model.enable_adapter_layers() # Re-enable adapters
            peft_model.train() # Switch back to train mode

            # --- Calculate DPO Loss ---
            # Ensure logits are gathered correctly if needed (DDP handles this for loss?)
            # Log probabilities should be calculated per sample.
            # HuggingFace model output includes loss, calculated based on logits vs labels.
            # Need log-probabilities of the chosen tokens (pos and neg sequences)

            # TODO: Verify DPO calculation correctness with HF output structure
            # This requires getting sequence probabilities, not just the cross-entropy loss.
            # Let's recalculate log probs from logits. Need to be careful with masking.

            def get_sequence_log_probs(logits, labels):
                # logits: (batch, seq_len, vocab_size)
                # labels: (batch, seq_len)
                log_probs = logits.log_softmax(dim=-1)
                # Gather log_probs corresponding to actual tokens (ignore padding -100)
                # Need to handle labels=-100. Shift labels? No, gather directly.
                valid_labels = labels.clone()
                valid_labels[labels == mask_token] = 0 # Replace -100 with a valid index (0) for gather
                gathered_log_probs = log_probs.gather(dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
                # Zero out log_probs where label was masked
                gathered_log_probs[labels == mask_token] = 0.0
                # Sum log probs for each sequence in the batch
                return gathered_log_probs.sum(dim=-1) # (batch_size,)

            log_probs_pos = get_sequence_log_probs(outputs_pos.logits, tokens_pos["labels"])
            log_probs_neg = get_sequence_log_probs(outputs_neg.logits, tokens_neg["labels"])
            ref_log_probs_pos = get_sequence_log_probs(ref_outputs_pos.logits, tokens_pos["labels"])
            ref_log_probs_neg = get_sequence_log_probs(ref_outputs_neg.logits, tokens_neg["labels"])

            beta = hparams.beta
            pi_logratios = log_probs_pos - log_probs_neg # log(pi_policy(pos)/pi_policy(neg))
            ref_logratios = ref_log_probs_pos - ref_log_probs_neg # log(pi_ref(pos)/pi_ref(neg))

            # DPO loss per sample
            dpo_advantage = beta * (pi_logratios - ref_logratios)
            dpo_loss = -torch.nn.functional.logsigmoid(dpo_advantage).mean() # Average over batch

            # Optional: Include standard language modeling loss (from policy model)
            # Use the loss returned by the model (it handles masking internally)
            lm_loss_pos = outputs_pos.loss # Use the pre-computed loss if available and correct
            lm_loss_neg = outputs_neg.loss
            # Should we average pos and neg LM loss? Or just use one? Or rely on DPO only?
            # If using LM loss, alpha controls weighting.
            # Let's use the positive sample's LM loss as proxy.
            lm_loss = lm_loss_pos

            # --- Combine Losses ---
            alpha = hparams.alpha
            # If alpha is 1.0, only use DPO loss. If 0.0, only LM loss.
            # The formula used in original code was `alpha * lora_loss + (1 - alpha) * dpo_loss`
            # Let's keep that structure, assuming lora_loss meant LM loss here.
            loss = alpha * lm_loss + (1 - alpha) * dpo_loss
            # loss = dpo_loss # Or just DPO loss if alpha = 1.0

            # --- Backward Pass & Optimization ---
            loss.backward() # DDP handles gradient synchronization automatically
            optimizer.step()

            # --- Logging ---
            # Average loss across all GPUs for logging
            batch_loss = loss.item() # Loss on this GPU's microbatch
            # Use a tensor for reduction
            batch_loss_tensor = torch.tensor(batch_loss, device=device)
            dist.all_reduce(batch_loss_tensor, op=dist.ReduceOp.AVG)
            avg_batch_loss = batch_loss_tensor.item()

            # Update meter only on rank 0 with the averaged loss
            if rank == 0:
                # The batch size used for update should be the global batch size
                global_batch_size = len(txt_batch) * world_size
                loss_meter.update(avg_batch_loss, n=len(txt_batch)) # len(txt_batch) is per-GPU batch size

            if rank == 0 and step % 10 == 0: # Print periodically from rank 0
                print(f"  Step: {step}/{len(dataloader)}, Avg Loss: {loss_meter.avg:.4f} (Current Batch Avg: {avg_batch_loss:.4f})")

        # End of Epoch
        if rank == 0:
            print(f"Epoch {epoch} finished. Average Loss: {loss_meter.avg:.4f}")

        # Optional: Add checkpoint saving logic here (save only on rank 0)
        # if rank == 0:
        #    save_path = f"./model_epoch_{epoch}.pt"
        #    # Save the unwrapped model's state dict
        #    torch.save(peft_model.state_dict(), save_path)
        #    print(f"Checkpoint saved to {save_path}")
        # dist.barrier() # Ensure all ranks finish before next epoch or saving


    # Return the underlying PEFT model state (weights are synchronized)
    return peft_model

# Example Usage (requires launching with torchrun)
if __name__ == "__main__":
    # This block will run on all processes launched by torchrun

    # --- 1. Setup DDP (already called inside apply_dpo_to_model) ---
    # setup_distributed() # Called within apply_dpo...
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f'cuda:{local_rank}')

    # --- 2. Load Model and Tokenizer (Load on CPU first if large, then move) ---
    # Example: Using a smaller model for demonstration
    model_name = "gpt2" # Replace with your actual model (e.g., "meta-llama/Llama-2-7b-hf")
    if rank == 0:
        print(f"Loading model {model_name}...")
    # Load base model (consider low_cpu_mem_usage if needed)
    model = AutoModelForCausalLM.from_pretrained(model_name) #.to(device) Model moved in apply_dpo...
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token # Common practice

    # --- 3. Prepare Dummy Data ---
    requests = [
        {"prompt": "What is the capital of France?", "target": "Paris is the capital.", "targets_neg": "Lyon is the capital.", "image": None},
        {"prompt": "Translate 'hello' to Spanish:", "target": "Hola", "targets_neg": "Bonjour", "image": None},
        # Add more data... Needs enough data for batching across GPUs
        {"prompt": "2 + 2 = ", "target": "4", "targets_neg": "5", "image": None},
        {"prompt": "The sky is", "target": "blue", "targets_neg": "green", "image": None},
        {"prompt": "Who wrote Hamlet?", "target": "Shakespeare", "targets_neg": "Dickens", "image": None},
        {"prompt": "Largest planet?", "target": "Jupiter", "targets_neg": "Mars", "image": None},
    ] * 10 # Make data larger for demonstration

    # --- 4. Hyperparameters ---
    hparams = DPOHyperParams()
    # Adjust hparams if needed, e.g., smaller batch size per GPU
    hparams.batch_size = 2 # Per GPU
    hparams.num_steps = 2 # Fewer epochs for demo
    hparams.target_modules = ["c_attn"] # Adjust for GPT2 target modules

    # --- 5. Run DPO Training ---
    if rank == 0:
        print("Starting DPO training function...")

    # Ensure model is on the correct device before passing it
    # model.to(device) # This is now handled inside apply_dpo_to_model

    trained_model, _ = apply_dpo_to_model(model, tok, requests, hparams)

    # --- 6. Post-Training (Optional: Save model on rank 0) ---
    if rank == 0:
        print("Training complete. Example of saving model...")
        # If PEFT was applied to a sub-module (like model.llava_model)
        if hasattr(trained_model, 'llava_model') and hasattr(trained_model.llava_model, 'save_pretrained'):
             trained_model.llava_model.save_pretrained("./dpo_trained_peft_adapter")
             print("PEFT adapter saved to ./dpo_trained_peft_adapter")
        # If PEFT was applied directly to the model and it's a PeftModel
        elif hasattr(trained_model, 'save_pretrained'):
             trained_model.save_pretrained("./dpo_trained_peft_adapter")
             print("PEFT adapter saved to ./dpo_trained_peft_adapter")
        else:
             print("Model structure doesn't seem to be a PEFT model, cannot save adapter directly.")
             # torch.save(trained_model.state_dict(), "./dpo_trained_model_full.pt") # Save full state dict if needed

    # Cleanup is handled within apply_dpo_to_model after the training call returns
    # cleanup_distributed() # Called within apply_dpo...

    if rank == 0:
        print("Script finished.")