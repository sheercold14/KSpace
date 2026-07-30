import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

def get_model_config(model, attribute_name):
        for sub_model_name in ['llama_model', 'opt_model', 'llava_model', 'qwen_model', 'phi_model', '']:
            sub_model = getattr(model, sub_model_name, model if sub_model_name == '' else None)
            if sub_model and hasattr(sub_model, 'config') and hasattr(sub_model.config, attribute_name):
                return getattr(sub_model.config, attribute_name)
        return None
    
def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
        )

        if get_model_config(model, 'n_positions'):
            maxlen = get_model_config(model, 'n_positions')
        elif get_model_config(model, 'max_sequence_length'):
            maxlen = get_model_config(model, 'max_sequence_length')
        elif get_model_config(model, 'max_position_embeddings'):
            maxlen = get_model_config(model, 'max_position_embeddings')
        elif get_model_config(model, 'seq_length'):
            maxlen = get_model_config(model, 'seq_length')
        else:
            raise NotImplementedError
        
        if get_model_config(model, 'model_type') and 'mistral' in get_model_config(model, 'model_type'):
            if get_model_config(model, 'sliding_window'):
                maxlen = get_model_config(model, 'sliding_window') or 4096
            else:
                maxlen = 4096
        if get_model_config(model, 'model_type') and 'qwen2' in get_model_config(model, 'model_type'):
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
 
    if get_model_config(model, 'n_positions'):
        npos = get_model_config(model, 'n_positions')
    elif get_model_config(model, 'max_sequence_length'):
        npos = get_model_config(model, 'max_sequence_length')
    elif get_model_config(model, 'max_position_embeddings'):
        npos = get_model_config(model, 'max_position_embeddings')
    elif get_model_config(model, 'seq_length'):
        npos = get_model_config(model, 'seq_length')
    else:
        raise NotImplementedError
        
    if get_model_config(model, 'model_type') and 'mistral' in get_model_config(model, 'model_type'):
        if get_model_config(model, 'sliding_window'):
            npos = get_model_config(model, 'sliding_window') or 4096
        else:
            npos = 4096
    if get_model_config(model, 'model_type') and 'qwen2' in get_model_config(model, 'model_type'):
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        if get_model_config(model,'_name_or_path'):
            model_name = get_model_config(model,'_name_or_path').rsplit("/")[-1]


    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if (not filename.exists() or force_recompute)  else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    batch_dict = {"text_input":[tokenizer.decode(batch['input_ids'][0][:200])],
                                  "image":None}
                    model(batch_dict)
                    #model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"][0][:200])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat
from easyeditor import VQADataset_Simple
def layer_stats_multimodal(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
    template=None
):
    """
    Function to load or compute cached stats.
    """
    def get_VQA_ds(prompt,template):
        annotation_path = hparams.train_annotation_path
        image_root = hparams.coco_image
        raw_ds = VQADataset_Simple(prompt=prompt,template=template,annotation_file=annotation_path,image_root=image_root,image_size=336)
        return raw_ds
    # Continue with computation of statistics
    batch_size = 1  # Examine this many dataset texts at once

    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        if get_model_config(model,'_name_or_path'):
            model_name = get_model_config(model,'_name_or_path').rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}_{sample_size}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    prompt = "{}"
    if hparams.model_name == 'llava':
        from ...trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
        prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
    ds = get_VQA_ds(prompt,template) if (not filename.exists() or force_recompute) else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        #collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    # batch_count = -(-(sample_size or len(ds)) // batch_size)
    batch_count = 1
    with torch.no_grad():
        for batch in progress(loader, total=batch_count):
            # batch = dict_to_(batch, f"cuda:{hparams.device}")
            with Trace(
                model, layer_name, retain_input=True, retain_output=False, stop=True
            ) as tr:
                model(batch)
            # feats = flatten_masked_batch(tr.input, batch["attention_mask"])
            # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
            feats = tr.input.to(dtype=dtype).squeeze()
            stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
