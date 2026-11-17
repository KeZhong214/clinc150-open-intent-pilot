"""Streamlit demo for Gate-3 open-intent routing.

Run from the project root:
    streamlit run app.py --server.headless true --server.port 8501
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings" / "gate3"
MODEL_CACHE = PROJECT_ROOT / "data" / "model_cache"
RESULTS = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate3.methods import (  # noqa: E402
    knn_scores,
    make_scores,
    prototypes_multi,
    prototypes_single,
    scores_from_prototypes,
)

DATASETS = {
    "banking77": "BANKING77 · 40 known banking intents",
    "massive": "MASSIVE en-US · 35 known assistant intents",
    "clinc150": "CLINC150 · 20 historical known intents",
}
MODELS = {
    "bge": {
        "id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "prompt": "",
        "device": "cpu",
    },
    "gte": {
        "id": "Alibaba-NLP/gte-modernbert-base",
        "revision": "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        "prompt": "",
        "device": "cpu",
    },
    "e5": {
        "id": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "prompt": "query: ",
        "device": "cpu",
    },
    "qwen": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "prompt": "Instruct: Given a customer service message, identify the supported intent it expresses.\nQuery: ",
        "device": "cuda",
    },
}
METHODS = ["B0", "B3-P", "B3-H"]
METHOD_HELP = {
    "B0": "single prototype + global threshold",
    "B3-P": "multi-prototype for high-dispersion classes + global threshold",
    "B3-H": "0.5 × B3-P score + 0.5 × top-2 support-neighbor score",
}
EXAMPLES = {
    "banking77": [
        "I still have not received my new card.",
        "How do I locate my card?",
        "I am still waiting on my card?",
    ],
    "massive": [
        "wake me up at five am",
        "turn up the volume",
        "what is the weather in london",
    ],
    "clinc150": [
        "I need to change my insurance policy",
        "what's your designation",
        "how much has the dow changed today",
    ],
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_key: str):
    spec = MODELS[model_key]
    device = spec["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    kwargs = {
        "device": device,
        "cache_folder": str(MODEL_CACHE),
        "revision": spec["revision"],
    }
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16, "attn_implementation": "sdpa"}
    model = SentenceTransformer(spec["id"], **kwargs)
    model.max_seq_length = 128
    model.eval()
    return model


@st.cache_resource
def load_state(model_key: str, dataset: str, shot: int = 5):
    info = read_json(MANIFEST / f"gate3_{dataset}_dataset.json")
    classes = info["known_classes"]
    support_path = MANIFEST / f"gate3_{dataset}_support_{shot}.csv"
    support = list(csv.DictReader(support_path.open("r", encoding="utf-8")))
    path = CACHE / model_key / dataset / MODELS[model_key]["revision"] / (
        f"L128_{hashlib.sha256(MODELS[model_key]['prompt'].encode()).hexdigest()[:16]}"
    ) / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    ids = data["sample_ids"].astype(str)
    idx = {sid: i for i, sid in enumerate(ids)}
    support_idx = [idx[row["sample_id"]] for row in support]
    X = data["vectors"][support_idx].astype(np.float32)
    y = np.asarray([row["intent"] for row in support])
    thresholds = read_json(RESULTS / "gate3_validation_thresholds.json")
    taus = {method: float(thresholds[f"{dataset}|{model_key}|{shot}|{method}"]["tau"]) for method in METHODS}
    details = read_json(RESULTS / "gate3_method_details.json")
    return classes, X, y, taus, details


def compute(model_key: str, dataset: str, method: str, text: str):
    classes, support_X, support_y, taus, details = load_state(model_key, dataset, 5)
    model = load_model(model_key)
    spec = MODELS[model_key]
    with torch.inference_mode():
        vec = model.encode(
            [text],
            prompt=spec["prompt"] or None,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    if method == "B0":
        protos = prototypes_single(support_X, support_y, classes)
        sims = scores_from_prototypes(vec, [p[None, :] for p in protos])
        meta = {}
    elif method == "B3-P":
        protos, meta = prototypes_multi(support_X, support_y, classes)
        sims = scores_from_prototypes(vec, protos)
    elif method == "B3-H":
        protos, meta = prototypes_multi(support_X, support_y, classes)
        p = scores_from_prototypes(vec, protos)
        n = knn_scores(vec, support_X, support_y, classes, top_k=2)
        sims = 0.5 * p + 0.5 * n
    else:
        raise ValueError(method)
    order = np.argsort(-sims[0])[:3]
    top3 = [(classes[i], float(sims[0, i])) for i in order]
    tau = taus[method]
    accepted = top3[0][1] >= tau
    return {
        "top3": top3,
        "tau": tau,
        "accepted": accepted,
        "pred": top3[0][0] if accepted else "OOS",
        "meta": meta,
    }


def main() -> None:
    st.set_page_config(page_title="Open Intent Routing", layout="wide")
    st.title("Open Intent Routing · Gate 3 Demo")
    st.caption("Real local encoders + support caches. The rejection threshold is the validation-selected Gate-3 threshold for shot=5.")

    with st.sidebar:
        dataset = st.selectbox("Dataset", list(DATASETS), format_func=lambda key: DATASETS[key])
        model_key = st.selectbox("Encoder", list(MODELS), format_func=lambda key: key.upper())
        method = st.radio("Method", METHODS, horizontal=False)
        st.caption(METHOD_HELP[method])
        st.divider()
        st.write("Available known classes:", len(load_state(model_key, dataset, 5)[0]))
        st.caption("Qwen uses CUDA when available; other encoders use CPU for the demo.")

    text = st.text_area("English customer utterance", value=EXAMPLES[dataset][0], height=90)
    run = st.button("Classify", type="primary") or st.query_params.get("demo") == "1"
    st.caption("The example is only a UI seed; the answer is computed from the selected encoder and cached support vectors.")

    if run:
        try:
            result = compute(model_key, dataset, method, text)
            st.subheader("Decision")
            if result["accepted"]:
                st.success(f"Accepted → {result['pred']}")
            else:
                st.warning("Rejected → OOS / unknown")
            st.write(f"Top-1 score **{result['top3'][0][1]:.4f}** · threshold **{result['tau']:.4f}**")
            st.dataframe(
                pd.DataFrame(
                    [{"rank": i + 1, "intent": name, "score": round(score, 5)} for i, (name, score) in enumerate(result["top3"])]
                ),
                use_container_width=True,
                hide_index=True,
            )
            if method == "B3-H":
                st.info(
                    "B3-H score = 0.5 × B3-P prototype score + 0.5 × mean top-2 support-neighbor cosine. "
                    "Reject happens when the fused Top-1 score is below the validation threshold."
                )
            elif method == "B3-P":
                st.info("B3-P uses 2 KMeans prototypes for the top-25% support-dispersion classes; other classes use one prototype.")
            else:
                st.info("B0 uses one normalized mean prototype per known class and a global validation threshold.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Demo computation failed: {type(exc).__name__}: {exc}")
            st.exception(exc)

    st.divider()
    st.markdown(
        "**Protocol:** support is frozen seed=42 / 5-shot; known/dev-unknown calibration "
        "selects the threshold; the demo does not use hidden test labels. "
        "BANKING77 and MASSIVE test splits were opened once by the Gate-3 matrix."
    )


if __name__ == "__main__":
    main()
