# KSpace: Knowledge Editing for Multimodal LLMs from Decomposed Subspaces

PyTorch implementation of KSpace, a parameter-efficient method for
multimodal knowledge editing. KSpace initializes the frozen LoRA matrix in
the null space of preserved activations and projects
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


## Acknowledgements

This codebase builds on
[EasyEdit](https://github.com/zjunlp/EasyEdit).
## License

The code is released under the MIT License. See [LICENSE](LICENSE).
