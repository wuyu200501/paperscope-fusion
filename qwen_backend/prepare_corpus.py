"""QASPER v0.3 -> cleaned chunks for retrieval (strategies A and B).

Used both as a CLI (build chunks_*.jsonl + print stats) and as a library
(the eval scripts import clean_text / iter_chunks so evidence-to-chunk
alignment uses exactly the same cleaning as ingestion).

Cleaning rules
  BIBREF{n}        -> ""        citation markers are noise for embeddings
  SECREF{n}        -> ""
  TABREF{n}        -> " Table"  keep readable wording ("Table 4 shows ..." style)
  FIGREF{n}        -> " Figure"
  INLINEFORM{n}    -> " [MATH]" S2ORC inline-formula placeholders
  DISPLAYFORM{n}   -> " [MATH]"
  whitespace normalized; empty paragraphs dropped;
  references/bibliography sections dropped (citation-list noise).

Strategy A (structured, word-based)
  unit = paragraph, prefixed with its section name;
  consecutive short paragraphs merged until >= MIN_WORDS;
  chunks exceeding MAX_WORDS split (no overlap);
  never crosses a section boundary; figure/table captions become
  their own chunks under section "[Figure/Table]".

Strategy B (fixed sliding window, token-based)
  whole paper serialized (title + sections + captions), then a
  512-token window with 64-token overlap, using the Qwen3 embedding
  tokenizer so token counts match the embedding model exactly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

MIN_WORDS = 100   # strategy A: merge consecutive paragraphs below this
MAX_WORDS = 450   # strategy A: hard cap, split above this
WIN_TOKENS = 512  # strategy B: window size
OVERLAP_TOKENS = 64
DROP_SECTION_RE = re.compile(r"^(references|bibliography)\s*$", re.I)

_BIBREF_RE = re.compile(r"\s?BIBREF\d+")
_SECREF_RE = re.compile(r"\s?SECREF\d+")
_TABREF_RE = re.compile(r"\s?TABREF\d+")
_FIGREF_RE = re.compile(r"\s?FIGREF\d+")
_INFORM_RE = re.compile(r"\s?INLINEFORM\d+")
_DISFORM_RE = re.compile(r"\s?DISPLAYFORM\d+")
_WS_RE = re.compile(r"[ \t]+")


def clean_text(raw: str) -> str:
    """Normalize one paragraph/caption. Shared by ingestion and eval."""
    s = raw.replace("\t", " ")
    s = _BIBREF_RE.sub("", s)
    s = _SECREF_RE.sub("", s)
    s = _TABREF_RE.sub(" Table", s)
    s = _FIGREF_RE.sub(" Figure", s)
    s = _INFORM_RE.sub(" [MATH]", s)
    s = _DISFORM_RE.sub(" [MATH]", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _raw_corpus_text(paper: dict) -> str:
    """Un-cleaned section paragraphs + captions (for marker counting)."""
    parts: list[str] = []
    for sec in paper.get("full_text", []):
        parts.extend(p or "" for p in sec.get("paragraphs", []))
    parts.extend(o.get("caption") or "" for o in paper.get("figures_and_tables", []))
    return " ".join(parts)


def _serialize_paper(paper: dict) -> str:
    """Whole-paper text for strategy B (sections + captions, no references)."""
    parts = [clean_text(paper.get("title") or "")]
    for sec in paper.get("full_text", []):
        name = (sec.get("section_name") or "").strip()
        if DROP_SECTION_RE.match(name):
            continue
        body = " ".join(clean_text(p) for p in sec.get("paragraphs", []) if p.strip())
        if body:
            parts.append(f"{name}. {body}" if name else body)
    for obj in paper.get("figures_and_tables", []):
        cap = clean_text(obj.get("caption", ""))
        if cap:
            parts.append(f"[Figure/Table] {cap}")
    return "\n\n".join(parts)


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, max_words: int) -> list[str]:
    """Split an over-long paragraph at sentence boundaries (<= max_words each);
    pathological unpunctuated spans fall back to a hard word-level split."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    pieces: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for sent in _SENT_RE.split(text):
        while len(sent.split()) > max_words:  # single sentence > cap: hard split
            head = " ".join(sent.split()[:max_words])
            sent = " ".join(sent.split()[max_words:])
            pieces.append(head)
        sw = len(sent.split()) if sent else 0
        if not sent:
            continue
        if buf and buf_words + sw > max_words:
            pieces.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += sw
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunks_strategy_a(paper: dict, paper_id: str) -> list[dict]:
    raw: list[tuple[str, str]] = []

    def add(section: str, text: str):
        text = text.strip()
        if text:
            raw.append((section, text))

    for sec in paper.get("full_text", []):
        name = (sec.get("section_name") or "").strip()
        if DROP_SECTION_RE.match(name):
            continue
        prefix = f"{name}. " if name else ""
        buf: list[str] = []
        buf_words = 0
        for para in sec.get("paragraphs", []):
            para = clean_text(para)
            if not para:
                continue
            for piece in _split_long(para, MAX_WORDS):
                w = len(piece.split())
                if buf and buf_words + w > MAX_WORDS:
                    add(name, prefix + " ".join(buf))
                    buf, buf_words = [], 0
                buf.append(piece)
                buf_words += w
                if buf_words >= MIN_WORDS:
                    add(name, prefix + " ".join(buf))
                    buf, buf_words = [], 0
        if buf:
            add(name, prefix + " ".join(buf))

    for obj in paper.get("figures_and_tables", []):
        cap = clean_text(obj.get("caption", ""))
        if cap:
            add("[Figure/Table]", f"[Figure/Table] {cap}")

    return [
        {
            "chunk_id": f"A:{paper_id}:{i}",
            "paper_id": paper_id,
            "strategy": "A",
            "section_name": section,
            "n_words": len(text.split()),
            "text": text,
        }
        for i, (section, text) in enumerate(raw)
    ]


def chunks_strategy_b(paper: dict, paper_id: str, tok) -> list[dict]:
    ids = tok(_serialize_paper(paper), add_special_tokens=False)["input_ids"]
    out: list[dict] = []
    step = WIN_TOKENS - OVERLAP_TOKENS
    prev_end = -1
    start = 0
    while start < len(ids):
        window = ids[start:start + WIN_TOKENS]
        end = start + len(window)
        if end <= prev_end:
            break  # final window fully covered by the previous one
        text = tok.decode(window).strip()
        if text:
            out.append({
                "chunk_id": f"B:{paper_id}:{len(out)}",
                "paper_id": paper_id,
                "strategy": "B",
                "section_name": "(sliding window)",
                "n_tokens": len(window),
                "text": text,
            })
            prev_end = end
        if end >= len(ids):
            break
        start += step
    return out


def load_papers(path: Path, limit: int | None = None) -> dict:
    if path.suffix in {".tgz", ".gz"}:
        with tarfile.open(path, "r:gz") as archive:
            source = next(
                member for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == "qasper-dev-v0.3.json"
            )
            with archive.extractfile(source) as stream:
                papers = json.load(stream)
    else:
        papers = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        papers = dict(list(papers.items())[:limit])
    return papers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/qasper-src/qasper-dev-v0.3.json"))
    ap.add_argument("--out", type=Path, default=Path("data/chunks"))
    ap.add_argument("--limit", type=int, default=None, help="only first N papers (debug)")
    args = ap.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
        print("tokenizer: Qwen/Qwen3-Embedding-4B loaded")
    except Exception as e:  # tokenizer is only needed for strategy B
        print(f"tokenizer unavailable ({e.__class__.__name__}); strategy B skipped", file=sys.stderr)

    papers = load_papers(args.data, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)

    stats = {"papers": len(papers), "bibref_removed": 0, "sections_total": 0}
    rows_a: list[dict] = []
    rows_b: list[dict] = []

    for pid, paper in papers.items():
        stats["bibref_removed"] += len(_BIBREF_RE.findall(_raw_corpus_text(paper)))
        stats["sections_total"] += len(paper.get("full_text", []))
        rows_a.extend(chunks_strategy_a(paper, pid))
        if tok is not None:
            rows_b.extend(chunks_strategy_b(paper, pid, tok))

    if tok is not None:  # exact token counts for strategy A as well
        for i in range(0, len(rows_a), 2048):
            batch = [r["text"] for r in rows_a[i:i + 2048]]
            for r, ids in zip(rows_a[i:i + 2048], tok(batch, add_special_tokens=False)["input_ids"]):
                r["n_tokens"] = len(ids)

    for name, rows in (("A", rows_a), ("B", rows_b)):
        path = args.out / f"chunks_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        words = sorted(len(r["text"].split()) for r in rows)
        toks = sorted(r.get("n_tokens", 0) or int(len(r["text"].split()) * 1.3) for r in rows)
        n = len(rows)
        print(f"\nstrategy {name}: {n} chunks -> {path}")
        if n:
            print(f"  words/chunk: mean={sum(words)/n:.0f} p50={words[n//2]} max={words[-1]}"
                  f" | tokens/chunk: mean={sum(toks)/n:.0f} p50={toks[n//2]} max={toks[-1]}")

    print(f"\ncorpus: {stats['papers']} papers, {stats['sections_total']} sections,"
          f" {stats['bibref_removed']} BIBREF markers removed")


if __name__ == "__main__":
    main()
