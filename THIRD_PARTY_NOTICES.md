# Third-party notices and attribution

This repository contains course-project code and derived manifests. It does
**not** contain model weights.

## CLINC150 / OOS-Eval dataset

- Source: <https://github.com/clinc/oos-eval>
- Pinned revision: `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- Data file: `data_full.json`
- License: **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**
- The official `LICENSE` file is included at `data/raw/LICENSE`.
- The raw dataset file in `data/raw/data_full.json` is included unmodified.
  No rows were deleted, rewritten, translated, or relabeled.
- Citation: Larson et al., "An Evaluation Dataset for Intent Classification
  and Out-of-Scope Prediction", EMNLP-IJCNLP 2019.
  <https://aclanthology.org/D19-1131/>

## BAAI/bge-small-en-v1.5

- Source: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- Revision used for the smoke test:
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- License: **MIT**
- Only the model card/reference files are included under
  `data/raw/model_cards/`; model weights and the local Hugging Face cache are
  not committed.

## Qwen/Qwen3-Embedding-0.6B

- Source: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- Revision checked: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- License: **Apache-2.0**
- Gate 0 did **not** download or load these weights.
- Only the model card/reference files are included under
  `data/raw/model_cards/`; no Qwen weights are committed.

## Python dependencies

Dependency versions are recorded in `requirements_lock.txt` and
`pip_freeze.txt`. Each package keeps its own license; this repository does not
relicense those packages.
