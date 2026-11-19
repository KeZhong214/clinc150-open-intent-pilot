# OpenIntent course clean package

This is the audited course-submission source snapshot of:

**少样本开放意图识别与智能客服：从开放意图拒识到可执行银行客服 Agent**

## Runtime

- Python environment: the original project `.venv` (Python 3.14, CUDA torch already installed).
- Models are loaded from the local Hugging Face cache under `data/model_cache` with
  `HF_HUB_OFFLINE=1`; base Qwen/GTE weights are **not** duplicated inside this ZIP.
- The selected small heads, Qwen Plain42 adapter, BANKING77 support manifest and
  the three selected embedding caches are included under `results/` and
  `data/cache_embeddings/`.

## Quick check

```powershell
$env:HF_HOME = (Join-Path $PWD 'data/model_cache')
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe preflight_demo.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_demo_release.py" -v
```

## Included experiment protocol

- Gate3: three datasets × four encoders × 2 shots × 3 methods = 72 configurations;
  P04 is the **B0 / 10-shot 12-cell slice**, not the full leaderboard.
- Gate5: B4 module attribution on BANKING77 GTE Frozen development validation.
- Gate7: 36 paired B4-head runs (6 encoder configurations × 2 methods × 3 seeds);
  displayed spread is ±1 sample standard deviation, not a 95% CI.
- Gate8.2: v1 original-trace corrected TSR = 60.00 / 60.00 / 70.83%;
  v2 (63.33/63.33/74.17) is a test-metadata regression and is **not** the final score.
- API mode was not re-evaluated on the 120-task benchmark and is not claimed to
  improve TSR.

## Exclusions

`.venv`, `data/raw`, historical prediction dumps, old app variants, old screenshots,
session databases, API keys, and local logs are intentionally excluded. The
GitHub history and all original experiment records remain untouched.
