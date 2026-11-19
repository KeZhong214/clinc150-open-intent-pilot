# OpenIntent demo clean package (source-only)

This package is for the classroom demo and **does not include the Qwen/GTE base
model weights**. It is designed to run on the original Windows machine where
`data/model_cache` already contains those weights. Do not copy `.venv` or an
API key into this package.

## One-command start

```powershell
.\start_demo.ps1
```

or double-click `start_demo.cmd`. The script sets offline model variables, runs
`preflight_demo.py`, then starts `app_demo.py` on port 8509. If the package is
extracted outside the original workspace, set `OPENINTENT_PYTHON` to the
original `.venv\Scripts\python.exe`; the base model cache must also exist at the
original path or be linked into `data/model_cache`. The app is not started if
preflight fails.

## 60–90 second demonstration

1. Click **① 查询 CARD-1002**.
   Expected: intent `card_arrival`, accepted, score about 0.7766, threshold
   0.4168; tool `get_card_delivery_status` returns `in_transit` / `inactive`.
2. Click **② 取消 TXN-2002**.
   Expected: `cancel_transfer`, `pending`; confirmation required and DB still
   `pending`.
3. Type **确认** (or `确认。`) and Enter.
   Expected: `cancelled`; the audit event is recorded; DB changes to cancelled.
4. Optional fallback: **③ OOS 备用**.
   Expected: OOS handoff; no card/transfer money tool executed.

## Modes and fallback

- **离线模式（无 API）**: the fixed Qwen Plain42 router plus Chinese templates.
  The card phrase is `我的卡CARD-1002还没到` (tight ID boundary is intentional).
- **DeepSeek 增强（需 API）**: if a valid `DEEPSEEK_API_KEY` is configured, the
  Chinese input is translated first and the same router/tools run. Without a
  valid key, the screen shows the real blocked/401 reason; it never silently
  bypasses the router. Use the offline mode for the classroom.
- The model is fixed; do not switch models during the live 60–90 seconds.

## Reset

Click **重置演示（新会话 DB）**. This creates a new fictional session database;
CARD-1002 returns to `inactive/in_transit`, TXN-2002 returns to `pending`.
It never deletes source experiment files.

## Evidence and exclusions

- All accounts/cards/transfers are fictional.
- The included Chinese Demo screenshot is from the offline mode; it is not an
  API-success claim.
- Base model files and `.venv` are excluded because they are large/local.
- See `docs/DEMO_OPERATIONS.md` for the on-stage operation card.
