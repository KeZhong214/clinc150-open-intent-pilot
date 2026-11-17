"""Small validation-only check used to fix B2-v2's kappa and u_quantile.

This script reads only known validation, original oos_val and hard_oos_val.
It never loads hard_oos_test.  The final B2-v2 implementation is in
``src/run_gate2.py``; this file is kept to document how kappa=10 and
u_quantile=0.99 were chosen before the freeze.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_embeddings"
RESULTS = ROOT / "results"
REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PH = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"


def load(path):
    z = np.load(path, allow_pickle=False)
    return (
        z["vectors"].astype(np.float32),
        z["sample_ids"].astype(str),
        z["roles"].astype(str),
        (z["labels"] if "labels" in z else z["intents"]).astype(str),
    )


known = load(CACHE / "qwen" / REV / f"seed42_shot5_L128_{PH[:16]}" / "embeddings.npz")
hard = load(CACHE / "qwen" / REV / f"hard_oos_val_L128_{PH[:16]}" / "embeddings.npz")
selected = json.loads((RESULTS.parent / "data" / "manifest" / "selected_intents.json").read_text(encoding="utf-8"))
classes = [x["intent"] for x in selected["intents"]]
cidx = {x: i for i, x in enumerate(classes)}
kv, kid, krole, klab = known
hv, hid, hrole, hlab = hard
sm = krole == "support"
sv = kv[sm]
sl = klab[sm]
P = []
for c in classes:
    p = sv[sl == c].mean(0)
    P.append(p / np.linalg.norm(p))
P = np.stack(P)


def scores(V):
    S = V @ P.T
    order = np.argsort(-S, axis=1)
    pred = order[:, 0]
    s1 = S[np.arange(len(S)), pred]
    s2 = S[np.arange(len(S)), order[:, 1]]
    return S, pred, s1, s1 - s2


km = krole == "calibration_known"
om = krole == "calibration_oos"
Sk, pk, s1k, mk = scores(kv[km])
So, po, s1o, mo = scores(kv[om])
Sh, ph, s1h, mh = scores(hv)
ytrue = np.array([cidx[x] for x in klab[km]])
n_k, n_o, n_h = len(s1k), len(s1o), len(s1h)

# Combined OOS: original 100 + hard 1300.
s1_all_oos = np.concatenate([s1o, s1h])
m_all_oos = np.concatenate([mo, mh])
p_all_oos = np.concatenate([po, ph])


def evaluate_accept(ak, ao):
    ka = ak.mean()
    kr = (np.where(ak, np.array(classes)[pk], "OOS") == np.array(classes)[ytrue]).mean()
    return float(ka), float(kr), float((~ao).mean())


def objective(ka, oos_recall):
    return oos_recall if ka >= 0.90 else -1


taus = np.unique(np.quantile(s1k, np.linspace(0, 1, 101)))


def best_b0():
    best = None
    for tau in taus:
        ak = s1k >= tau
        ao = s1_all_oos >= tau
        ka, kr, rec = evaluate_accept(ak, ao)
        if ka >= 0.90:
            key = (rec, ka, tau)
            if best is None or key > best[0]:
                best = (key, tau, ka, kr, rec)
    return best


b0 = best_b0()
print("B0 recalibrated combined val:", b0)

# B2 original formula, old q_c = q10(s1 | true c), lambda grid, gamma=0 first.
q_old = np.array([np.quantile(s1k[ytrue == c], .1) for c in range(len(classes))])
gammas = np.unique(np.concatenate([[0.0], np.quantile(mk, np.linspace(0, .9, 19))]))
best = None
for lam in (0, .5, 1):
    for tau in taus:
        tc = (1 - lam) * tau + lam * q_old
        for gamma in gammas:
            ak = (s1k >= tc[pk]) & (mk >= gamma)
            ao = (s1_all_oos >= tc[p_all_oos]) & (m_all_oos >= gamma)
            ka, kr, rec = evaluate_accept(ak, ao)
            if ka >= .9:
                key = (rec, ka, -gamma, -lam, tau)
                if best is None or key > best[0]:
                    best = (key, lam, tau, gamma, ka, kr, rec)
print("B2-orig recalibrated with gamma:", best)

q_new = np.array([
    np.quantile(Sk[ytrue == c, c], .1) for c in range(len(classes))
])

# B2-v2 candidate: only raise global tau toward hard-OOS upper tail, cap at q_c.
def run_v2(kappa: int, uq: float):
    u = np.full(len(classes), np.nan)
    n = np.zeros(len(classes), dtype=int)
    for c in range(len(classes)):
        mask = ph == c
        n[c] = mask.sum()
        if n[c]:
            u[c] = np.quantile(s1h[mask], uq)
    best = None
    for tau in taus:
        cap = np.maximum(tau, q_new)
        target = np.where(np.isnan(u), tau, np.maximum(tau, np.minimum(u, cap)))
        alpha = n / (n + kappa)
        tc = (1 - alpha) * tau + alpha * target
        for gamma in gammas:
            ak = (s1k >= tc[pk]) & (mk >= gamma)
            ao = (s1_all_oos >= tc[p_all_oos]) & (m_all_oos >= gamma)
            ka, kr, rec = evaluate_accept(ak, ao)
            if ka >= .9:
                key = (rec, ka, -gamma, tau)
                if best is None or key > best[0]:
                    best = (key, tau, gamma, ka, kr, rec, tc)
    return best


for kappa in (10, 25, 50, 100, 200):
    for uq in (.80, .85, .90, .95, .99):
        b = run_v2(kappa, uq)
        if b:
            _, tau, gamma, ka, kr, rec, tc = b
            print(
                f"v2 kappa={kappa:3d} uq={uq:.2f} tau={tau:.4f} gamma={gamma:.4f} "
                f"ka={ka:.4f} kr={kr:.4f} oos={rec:.4f} "
                f"tc_range=({tc.min():.3f},{tc.max():.3f})"
            )
