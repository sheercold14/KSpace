import torch
from torch.utils.data import Dataset, DataLoader

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
        img = r.get("image", None)
        return {
            "text": text,
            "pos": pos,
            "neg": neg,
            "image": img
        }

def dpo_collate_fn(batch):
    """
    将一批 item (dict) 合并成一个 batch dict。
    """
    texts = [x["text"] for x in batch]
    pos_targets = [x["pos"] for x in batch]
    neg_targets = [x["neg"] for x in batch]
    images = [x["image"] for x in batch]  # 可能全是 None（纯文本场景），或张量/图像对象
    return {
        "texts": texts,
        "pos_targets": pos_targets,
        "neg_targets": neg_targets,
        "images": images,
    }

from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设这里就是你定义的 超参数类
from .dpo_hparams import DPOMultimodalHyperParams, DPOHyperParams

# 把上面写好的 DPODataset, dpo_collate_fn 导入或放一起
# from .my_dataset import DPODataset, dpo_collate_fn

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
    使用多卡加速(Accelerate)对给定模型执行DPO。
    1) 给 model.llava_model (或 model 本身) 添加 LoRA
    2) 创建 Dataset + DataLoader 包装 requests
    3) 调用 execute_dpo 进行多卡训练
    """

    # 可选: 如果需要复制/备份原权重
    weights_copy = {}
    if copy:
        pass

    device = torch.device(f'cuda:{hparams.device}')
    print(f"Using device: {device}")

    # ========== 1. 准备LoRA ==========

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules
    )

    # 如果是 LLaVa，有 .llava_model，就对它做 LoRA；否则对 model 自身
    target_peft_model = model.llava_model if hasattr(model, "llava_model") else model
    

    peft_model = get_peft_model(target_peft_model, peft_config).to(device)
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()


    # 只训练LoRA层
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # ========== 2. 构建Dataset + DataLoader ==========

    dataset = DPODataset(requests)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=dpo_collate_fn,
    )

    # ========== 3. 进入DPO训练循环 ==========

    # 注意，这里会返回修改后的 peft_model
    edited_llava_model = execute_dpo(
        model=model,
        peft_model=peft_model,
        tok=tok,
        dataloader=dataloader,
        hparams=hparams,
    )

    # 把LoRA好的语言模型替换回去
    if hasattr(model, "llava_model"):
        model.llava_model = edited_llava_model
    else:
        model = edited_llava_model

    return model, weights_copy

def execute_dpo(
        model: AutoModelForCausalLM,
        peft_model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        dataloader: DataLoader,
        hparams: DPOHyperParams,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    使用Accelerate包装的多卡训练循环。
    - dataloader 直接产出一个batch，包含 {texts, pos_targets, neg_targets, images}
    - 根据有无 images (多模态 vs 纯文本) 做 forward
    - DPO对比 & 反向传播
    """

    peft_model.train()

    # 优化器
    optimizer = Adam(peft_model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

    # 加速器
    accelerator = Accelerator()

    # 同时包装 模型 + 优化器 + 数据加载器
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    loss_meter = AverageMeter()

    for epoch in range(hparams.num_steps):
        print("=" * 30)
        print(f"Epoch: {epoch}")
        print("=" * 30)
        loss_meter.reset()

        for step, batch in enumerate(dataloader):
            # batch 是从 dpo_collate_fn 来的
            txt_batch = batch["texts"]
            tgt_pos_batch = batch["pos_targets"]
            tgt_neg_batch = batch["neg_targets"]
            img_batch = batch["images"]  # 可能全是None, 也可能是图像

            # 清零梯度
            optimizer.zero_grad()
            mask_token = -100

            # 1) 正向样本 forward
            if any(x is not None for x in img_batch): # 如果 batch 里至少有一个 非None 的图像，就走多模态
                full_prompt_pos = [f"{p} {l}" for p, l in zip(txt_batch, tgt_pos_batch)]
                samples_pos = {
                    "noise": True,
                    "text_input": full_prompt_pos,
                    "image": img_batch,
                    "train": True
                }
                # 这里假设 model(...) 或 peft_model(...) 能识别 {image, text_input} 结构
                outputs_pos = model(samples_pos, output_attentions=False)
            else:
                # 全是纯文本
                full_prompt_pos = [f"{p} {l}" for p, l in zip(txt_batch, tgt_pos_batch)]
                tokens_pos = tok(full_prompt_pos, return_tensors="pt", padding=True, truncation=True)
                # 注意: accelerate 不再需要 .to(device)
                tokens_pos["labels"] = tokens_pos["input_ids"].clone()
                tokens_pos["labels"][tokens_pos["input_ids"] == tok.pad_token_id] = mask_token
                outputs_pos = peft_model(**tokens_pos)

            # 2) 负向样本 forward
            if any(x is not None for x in img_batch):
                full_prompt_neg = [f"{p} {l}" for p, l in zip(txt_batch, tgt_neg_batch)]
                samples_neg = {
                    "noise": True,
                    "text_input": full_prompt_neg,
                    "image": img_batch,
                    "train": True
                }
                outputs_neg = model(samples_neg, output_attentions=False)
            else:
                full_prompt_neg = [f"{p} {l}" for p, l in zip(txt_batch, tgt_neg_batch)]
                tokens_neg = tok(full_prompt_neg, return_tensors="pt", padding=True, truncation=True)
                tokens_neg["labels"] = tokens_neg["input_ids"].clone()
                tokens_neg["labels"][tokens_neg["input_ids"] == tok.pad_token_id] = mask_token
                outputs_neg = peft_model(**tokens_neg)

            # 3) Reference forward (disable LoRA)
            peft_model.eval()
            peft_model.disable_adapter_layers()
            with torch.no_grad():
                if any(x is not None for x in img_batch):
                    ref_outputs_pos = model(samples_pos, output_attentions=False)
                    ref_outputs_neg = model(samples_neg, output_attentions=False)
                else:
                    ref_outputs_pos = peft_model(**tokens_pos)
                    ref_outputs_neg = peft_model(**tokens_neg)
            peft_model.train()
            peft_model.enable_adapter_layers()

            # 4) 计算 DPO Loss
            lora_loss = outputs_pos.loss
            beta = hparams.beta

            ref_log_probs_pos = ref_outputs_pos.logits.log_softmax(dim=-1)
            ref_log_probs_neg = ref_outputs_neg.logits.log_softmax(dim=-1)

            log_probs_pos = outputs_pos.logits.log_softmax(dim=-1)
            log_probs_neg = outputs_neg.logits.log_softmax(dim=-1)

            dpo_advantage = beta * (
                (log_probs_pos - ref_log_probs_pos).sum(-1)
                - (log_probs_neg - ref_log_probs_neg).sum(-1)
            )
            # dpo_loss = -torch.mean(torch.log(torch.sigmoid(dpo_advantage)))
            dpo_loss = -torch.nn.functional.logsigmoid(dpo_advantage).mean()

            # 5) 合并总loss
            alpha = hparams.alpha
            loss = alpha * lora_loss + (1 - alpha) * dpo_loss

            # 6) 反向传播 & 更新
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()

            bs = len(txt_batch)
            loss_meter.update(loss.item(), n=bs)

        print(f"Total loss {loss_meter.avg}")

    return peft_model
