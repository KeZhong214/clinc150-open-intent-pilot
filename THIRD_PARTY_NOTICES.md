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
- Revision used: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- License: **Apache-2.0**
- Weights were downloaded locally for Gate 1/2/3 but are **not committed**;
  only metadata and cached embeddings outside git are used.

## GTE-ModernBERT-base

- Source: <https://huggingface.co/Alibaba-NLP/gte-modernbert-base>
- Revision used: `e7f32e3c00f91d699e8c43b53106206bcc72bb22`
- License: **Apache-2.0**
- Model weights are not committed.

## E5-base-v2

- Source: <https://huggingface.co/intfloat/e5-base-v2>
- Revision used: `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`
- License: **MIT**
- Inputs use the model-card `query: ` prefix.
- Model weights are not committed.

## BANKING77

- Source: <https://huggingface.co/datasets/PolyAI/banking77>
- Revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- License: **CC BY 4.0**
- Raw parquet files are included under `data/gate3/raw/banking77/`.
- Citation: Casanueva et al., "Efficient Intent Detection with Dual Sentence
  Encoders", 2020. <https://aclanthology.org/2020.nlp4convai-1.5/>

## MASSIVE en-US

- Source: <https://huggingface.co/datasets/AmazonScience/massive>
- Revision: `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`
- Dataset version: 1.1
- License: **CC BY 4.0** (official LICENSE included under
  `data/gate3/raw/massive/1.1/LICENSE`)
- The en-US JSONL and citation/notice files are included; the 40 MB multilingual
  tar.gz archive is git-ignored.
- Citation: FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural
  Language Understanding Dataset with 51 Typologically-Diverse Languages",
  2022. <https://arxiv.org/abs/2204.08582>

## SetFit and demo dependencies

- SetFit: <https://github.com/huggingface/setfit>, **Apache-2.0**
- Streamlit: <https://streamlit.io/>, **Apache-2.0**
- Qwen3-Embedding-0.6B: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>, **Apache-2.0**
- PEFT: <https://github.com/huggingface/peft>, **Apache-2.0**
- Playwright (screenshot tooling only): **Apache-2.0**

## Python dependencies

Dependency versions are recorded in `requirements_lock.txt` and
`pip_freeze.txt`. Each package keeps its own license; this repository does not
relicense those packages.
