"""Cross-encoder reranking with Qwen3-Reranker.

Default backend: **llama-server** — the llama.cpp server binary that ships
inside the Ollama install (`lib/ollama/llama-server.exe`), launched with
`--rerank`. It exposes the Jina/Cohere-style `/v1/rerank` endpoint and runs
the GGUF on CUDA. No torch, no extra downloads.

  RERANK_URL=http://127.0.0.1:8012/v1/rerank   (0.6B) / :8013 (4B)
  RERANK_BACKEND=llamaserver | transformers

The transformers backend (fp16 safetensors, needs torch) is kept for
reference; the llama-server path is what the evaluations use.
"""
from __future__ import annotations

import json
import os
import urllib.request

BACKEND = os.environ.get("RERANK_BACKEND", "llamaserver")
RERANK_URL = os.environ.get("RERANK_URL", "http://127.0.0.1:8012/v1/rerank")
MODEL_ID = os.environ.get("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
INSTRUCTION = (
    "Given a user question about a scientific paper, retrieve the passages "
    "from that paper that help answer the question."
)


def _rerank_llamaserver(query: str, docs: list[str], timeout: int = 600) -> list[float]:
    """POST /v1/rerank; returns scores aligned to the input order."""
    payload = json.dumps({
        "query": query,
        "documents": docs,
        "top_n": len(docs),
        "instruction": INSTRUCTION,
    }).encode()
    req = urllib.request.Request(RERANK_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    if data.get("error"):
        raise RuntimeError(f"rerank error: {data['error']}")
    scores = [0.0] * len(docs)
    for item in data.get("results", []):
        scores[item["index"]] = float(item["relevance_score"])
    return scores


# ---------------------------------------------------------------- transformers
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_model = None
_tok = None


def _get_transformers_model():
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        ).cuda().eval()
    return _model, _tok


def _rerank_transformers(query: str, docs: list[str], batch_size: int = 8,
                         max_length: int = 8192) -> list[float]:
    import torch
    import torch.nn.functional as F

    model, tok = _get_transformers_model()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    prefix = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n"
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    pair_ids = [
        (prefix_ids + tok(f"<Document>: {d}", add_special_tokens=False)["input_ids"]
         + suffix_ids)[-max_length:]
        for d in docs
    ]
    scores: list[float] = []
    for i in range(0, len(pair_ids), batch_size):
        inputs = tok.pad({"input_ids": pair_ids[i:i + batch_size]},
                         padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        two = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
        scores.extend(F.log_softmax(two, dim=1)[:, 1].exp().tolist())
    return scores


def rerank(query: str, docs: list[str], **kw) -> list[float]:
    """Return one relevance score in (0, 1) per doc, higher = more relevant."""
    if not docs:
        return []
    if BACKEND == "transformers":
        return _rerank_transformers(query, docs, **kw)
    return _rerank_llamaserver(query, docs)
