# OpenIntent full demo project

Rich Streamlit Agent project for the classroom, not the minimal two-button
sample. The entry is the original full workbench:

```text
app_gate8.py
```

It includes:

- the full Chinese workbench: model/mode selector, all 9 demos, intent card,
  Top-3, tool timeline, KB citations, SQLite state, audit and session reset;
- the real GTE/Qwen paired routers and B4-RejectOnly business tools;
- the Chinese translation adapter and grounded DeepSeek writer;
- the 30-document fictional policy KB;
- selected model heads, Qwen Plain42 adapter and selected embedding caches;
- the final 13-page PPTX under `deliverables/`;
- tests, reports, the PPT analysis figures and the two-page operation card.

## API-first launch

The API key is placed only in the local, Git-ignored file:

```text
configs/demo_api.env
```

Create it from `configs/demo_api.env.example`, then run:

```powershell
cd <this folder>
.\start_full_demo.ps1
```

The launcher:

1. reads only `DEEPSEEK_*` settings without printing the key;
2. sets offline model variables;
3. runs `preflight_demo.py` and only starts the app on `PREFLIGHT_OK`;
4. launches the full `app_gate8.py` on `http://127.0.0.1:8506`;
5. preselects `DeepSeek-assisted` and `Qwen-Plain42-RejectOnly`.

The full app remains usable in `Offline Router-driven` mode when the API is
missing, invalid or disconnected. That fallback is explicit, never silent.

## Base-model caveat

The ZIP is **source-only** with respect to the base Qwen/GTE model weights:
they are large local caches under `data/model_cache` and are not duplicated.
The selected heads, adapter and NPZ caches are included. Run this package on
the original Windows machine (or link the original model cache).

## Security

`configs/demo_api.env` is included **only in this local ZIP**, is ignored by
Git, and must not be uploaded to GitHub, sent in chat, or shown in a screenshot.
The launcher and preflight never echo the key.
