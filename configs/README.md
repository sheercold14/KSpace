# Configuration notes

All paths are interpreted relative to the repository root when the supplied
scripts are launched from that directory.

- `kspace/llava_evqa.yaml` is the main LLaVA-v1.5-7B configuration.
- `kspace/qwen25vl_evqa.yaml` and `kspace/phi4mm_evqa.yaml` provide the
  cross-backbone settings.
- `baselines/rank128_evqa/` contains the rank-matched LLaVA comparison.

The historical `alg_name: XSpace` is required by the current EasyEdit
registry and corresponds to KSpace in the paper.

Online judging is disabled in all public YAML files. If enabled locally,
keep the credential file under `secrets/` and never commit it.
