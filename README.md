# KSpace: Knowledge Editing for Multimodal LLMs from Decomposed Subspaces

PyTorch implementation of KSpace, a parameter-efficient method for
multimodal knowledge editing. KSpace initializes the frozen LoRA matrix in
an approximate low-variance subspace of preserved activations and projects
the trainable LoRA updates onto an edit-relevant knowledge subspace.

The implementation is built on [EasyEdit](https://github.com/zjunlp/EasyEdit).
For compatibility with existing checkpoints, KSpace is registered as
`XSpace` in the source code and configuration files.

## Installation

Python 3.10 is recommended.

```bash
conda create -n kspace python=3.10 -y
conda activate kspace
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The experiments were tested with PyTorch 2.5.1 and CUDA 12.1. Install the
PyTorch build that matches the CUDA driver on your machine.

## Data preparation

Download E-VQA and MMKE-Bench from their original sources and place them
under `data/`. The expected directory layout is described in
[`data/README.md`](data/README.md).

Model checkpoints are loaded from Hugging Face by default:

- `liuhaotian/llava-v1.5-7b`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `microsoft/Phi-4-multimodal-instruct`

Local checkpoints can be used by changing `name`, `tokenizer_name`, and
`cache_dir` in the corresponding configuration file.

## Running KSpace

The following command checks the configuration without loading a model:

```bash
python scripts/run_kspace.py \
  --config configs/kspace/llava_evqa.yaml \
  --dry-run
```

Run one E-VQA edit with:

```bash
python scripts/run_kspace.py \
  --config configs/kspace/llava_evqa.yaml \
  --data-path data/editing-data/vqa/vqa_eval.json \
  --image-root data \
  --num-edits 1
```

Configurations for the other backbones are available in
`configs/kspace/`.

## Baselines

Rank-matched configurations for LoRA, AdaLoRA, RoseLoRA, CorDA, and
LoRA-Null are provided in `configs/baselines/rank128_evqa/`.

```bash
python scripts/run_baseline.py \
  --config configs/baselines/rank128_evqa/llava_lora_r128_evqa.yaml \
  --data-path data/editing-data/vqa/vqa_eval.json \
  --image-root data \
  --num-edits 1
```

All methods use the same multimodal editor and evaluation pipeline.

## Code structure

```text
easyeditor/
├── editors/                  multimodal editing pipeline
├── evaluate/                 evaluation functions
├── models/
│   ├── xspace/               KSpace
│   ├── loranull/             LoRA-Null initialization
│   ├── lora/                 LoRA and AdaLoRA
│   ├── roselora/             RoseLoRA
│   ├── rome/                 ROME
│   ├── memit/                MEMIT
│   ├── alphaedit/            AlphaEdit
│   ├── unke/                 UnKE
│   ├── unike/                UniKE
│   └── mmelo/                Multi-MELO
└── trainer/                  multimodal model wrappers

configs/                      experiment settings
scripts/                      training and evaluation entry points
```

The main KSpace implementation is in
`easyeditor/models/xspace/xspace_main.py`. The projected optimizer is in
`easyeditor/models/xspace/optim/adam_svd.py`.

## Acknowledgements

This codebase builds on
[EasyEdit](https://github.com/zjunlp/EasyEdit) and includes integrations of
several knowledge-editing and parameter-efficient fine-tuning methods. We
thank the authors of these projects for releasing their code.

## License

The code is released under the MIT License. See [LICENSE](LICENSE).
