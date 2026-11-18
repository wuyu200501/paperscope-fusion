import json
import os

import streamlit as st

from paperscope import DEEPSEEK_MODEL, ROOT, PaperScope, STRATEGIES, answer_with_api, load_papers
from qwen_backend.paperscope_adapter import STRATEGIES as QWEN_STRATEGIES
from qwen_backend.paperscope_adapter import QwenRetriever
from qwen_backend.retrieve import DEFAULT_HOST, DEFAULT_MODEL


st.set_page_config(page_title="PaperScope", layout="wide")
st.html("<style>[data-testid='stText'] {white-space: pre-wrap; overflow-wrap: anywhere;}</style>")


@st.cache_resource(show_spinner="Loading papers and building the vector index…")
def open_paperscope(embedding):
    return PaperScope(embedding=embedding, index_dir=os.getenv("PAPERSCOPE_QDRANT_DIR") or None)


@st.cache_resource(show_spinner="Loading the Qwen index and paper catalog…")
def open_qwen():
    database = ROOT / os.getenv("PAPERSCOPE_QWEN_DB", "data/chroma")
    retriever = QwenRetriever(
        database, host=os.getenv("PAPERSCOPE_OLLAMA_HOST", DEFAULT_HOST),
        model=os.getenv("PAPERSCOPE_QWEN_MODEL", DEFAULT_MODEL),
    )
    return retriever, load_papers(paper_limit=None)


st.title("PaperScope")
st.markdown("Ask about a research paper or check a statement against its source passages.")

page = st.radio("Workspace", ["Ask a question", "Check a statement", "Evaluation"],
                horizontal=True, label_visibility="collapsed")
if page == "Evaluation":
    st.header("Evaluation results")
    for embedding in ("small", "base"):
        live_path = ROOT / "data" / f"claim-ablation-{embedding}" / "metrics.json"
        published_path = ROOT / "experiments" / f"claim_ablation_{embedding}.json"
        metrics_path = live_path if live_path.is_file() else published_path
        if metrics_path.is_file():
            live = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.subheader(f"Statement-checking experiment · BGE-{embedding}")
            st.caption(f"{live['model']} · Top {live['top_k']} · run {live['fingerprint']}")
            st.dataframe([
                {
                    "Configuration": key,
                    "Cases": row["cases"],
                    "Accuracy": f"{row['accuracy']:.1%}",
                    "False acceptance": f"{row['false_acceptance_rate']:.1%}",
                    "Macro-F1": f"{row['macro_f1']:.3f}",
                    "Median latency (s)": round(row["median_seconds"], 2),
                    "Tokens": row["prompt_tokens"] + row["completion_tokens"],
                }
                for key, row in live["metrics"].items()
            ], hide_index=True, width="stretch")
    comparison_path = ROOT / "experiments" / "comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        st.subheader("Qwen reranking on matched questions")
        st.caption("Saved Qwen runs · 40 answerable questions per comparison · complete evidence-string matching")
        st.dataframe(comparison["qwen"], hide_index=True, width="stretch")
        st.download_button("Download paired comparison", comparison_path.read_bytes(),
                           file_name=comparison_path.name, mime="application/json")
    st.subheader("PaperScope benchmark")
    replay_path = ROOT / "PaperScope-Replay.html"
    if replay_path.is_file():
        st.download_button(
            "Download offline experiment replay",
            replay_path.read_bytes(),
            file_name=replay_path.name,
            mime="text/html",
        )
    report_path = ROOT / "results" / "metrics.json"
    if not report_path.exists():
        st.info("Evaluation results will appear here after the benchmark completes.")
        st.stop()
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    st.caption(f"{summary['model']} · top {summary['top_k']} passages · run {summary['fingerprint']}")
    st.subheader("Question answering")
    qa_rows, claim_rows = [], []
    for key, metrics in summary["metrics"].items():
        strategy, task = key.split("/")
        record = {
            "Chunking": STRATEGIES[strategy],
            "Cases": metrics["cases"],
            "Accuracy": f"{metrics['accuracy']:.1%}",
            "Median time": f"{metrics['median_seconds']:.2f} s",
        }
        if task == "qa":
            record.update({
                "Hit@5": f"{metrics['hit_at_5']:.1%}",
                "Recall@5": f"{metrics['recall_at_5']:.1%}",
                "Answerable span match": f"{metrics['answerable_span_accuracy']:.1%}",
                "Correct refusal": f"{metrics['correct_refusal_rate']:.1%}",
            })
            qa_rows.append(record)
        else:
            record.update({
                "Method": task.title(),
                "Supported preserved": f"{metrics['preservation_rate']:.1%}",
                "False acceptance": f"{metrics['false_acceptance_rate']:.1%}",
            })
            claim_rows.append(record)
    st.dataframe(qa_rows, hide_index=True, width="stretch")
    st.caption("QA accuracy uses a normalized reference-span match. Refusal is measured on four unrelated questions.")
    st.subheader("Statement checking")
    st.dataframe(claim_rows, hide_index=True, width="stretch")
    st.caption("Both checking methods receive the same retrieved passages for each chunking strategy.")
    st.download_button("Download metrics", data=report_path.read_bytes(),
                       file_name="paperscope-metrics.json", mime="application/json")
    st.stop()

st.sidebar.header("Search settings")
api_enabled = bool(os.getenv("PAPERSCOPE_API_KEY"))
backends = ["BGE-small", "BGE-base"]
if api_enabled:
    backends.append("Qwen")
backend = st.sidebar.radio("Retrieval backend", backends)
use_qwen = backend == "Qwen"
strategies = QWEN_STRATEGIES if use_qwen else STRATEGIES
strategy = st.sidebar.radio("Chunking", list(strategies), index=int(not use_qwen),
                            format_func=strategies.get)
rerank = use_qwen and st.sidebar.checkbox("Rerank top 24 candidates")
try:
    if use_qwen:
        scope, papers = open_qwen()
    else:
        scope = open_paperscope("base" if backend == "BGE-base" else "small")
        papers = scope.papers
        if scope.index_path:
            st.sidebar.caption("Persistent Qdrant index")
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Could not load the selected retriever ({type(error).__name__}).")
    st.info("Check the QASPER archive and the selected backend setup in the README.")
    st.stop()

checking = page == "Check a statement"
modes = ["DeepSeek API", "Local extractive QA"] if api_enabled else ["Local extractive QA"]
if checking and not api_enabled:
    st.info("Start PaperScope in DeepSeek mode to check statements.")
    st.stop()
mode = "DeepSeek API" if checking or use_qwen else st.sidebar.radio("Answer mode", modes)
method = st.sidebar.radio("Checking method", ("Checked", "Direct")).lower() if checking else "qa"
if mode == "Local extractive QA":
    st.sidebar.caption("Answers are short excerpts selected by a local model.")
else:
    st.sidebar.caption(f"{os.getenv('PAPERSCOPE_MODEL', DEEPSEEK_MODEL)} receives the input and retrieved excerpts.")
st.sidebar.caption(f"Paper catalog: {len(papers)} QASPER papers")

paper_id = st.selectbox(
    "Research paper", list(papers),
    format_func=lambda key: papers[key]["title"],
)
with st.form("statement_form" if checking else "question_form"):
    if checking:
        question = st.text_area(
            "Statement", max_chars=400, placeholder="Paste a claim about the selected paper…"
        )
    else:
        question = st.text_input("Question", max_chars=400, placeholder="Ask in English…")
    submitted = st.form_submit_button(
        "Check evidence" if checking else "Find an answer", type="primary"
    )

settings = (paper_id, backend, strategy, rerank, mode, page, method)
if submitted:
    st.session_state.pop("result", None)
    if not question.strip():
        st.warning("Enter a statement to check." if checking else "Enter a question to search this paper.")
    else:
        with st.spinner("Searching the paper…"):
            try:
                if use_qwen:
                    evidence = scope.retrieve(
                        paper_id, question.strip(), strategy, rerank_n=24 if rerank else 0,
                    )
                else:
                    evidence = scope.retrieve(paper_id, question.strip(), strategy)
                if mode == "DeepSeek API":
                    answer = answer_with_api(question.strip(), evidence, method)
                else:
                    answer = scope.answer_locally(question.strip(), evidence)
                st.session_state.result = (settings, question.strip(), answer, evidence)
            except Exception as error:
                st.error(f"The answer could not be generated ({type(error).__name__}).")

result = st.session_state.get("result")
if result and result[0] == settings:
    _, submitted_question, answer, evidence = result
    st.divider()
    left, right = st.columns([1, 1.3], gap="large")
    left.subheader("Verdict" if checking else "Answer")
    left.text(submitted_question)
    cited = {source_id for _, source_id in answer.lines}
    if checking:
        label = answer.verdict.title()
        if answer.verdict == "SUPPORTED":
            left.success(label)
        elif answer.verdict == "CONTRADICTED":
            left.error(label)
        else:
            left.warning("Insufficient evidence")
    if answer.citation_error:
        left.warning("The response was withheld because a citation could not be verified.")
    if answer.conditions:
        left.subheader("Condition-by-condition evidence")
        for row in answer.conditions:
            with left.expander(f"{row.condition} · {row.verdict}", expanded=True):
                st.text(f"Statement: {row.statement}")
                st.text(f"Evidence finding: {row.finding}")
                if row.source_id:
                    st.caption(row.source_id)
                    st.text(row.quote)
    elif cited:
        for index, (claim, source_id) in enumerate(answer.lines):
            left.text(claim)
            source = next(item for item in evidence if item["id"] == source_id)
            left.text(f"{source['section']} · {source_id}")
            if answer.citations:
                left.caption(f"“{answer.citations[index]}”")
    elif not checking and not answer.citation_error:
        left.warning("I couldn't find enough evidence in this paper to answer.")
    left.caption(mode)

    right.subheader("Source passages")
    if not cited:
        right.caption("Retrieved for inspection. The passages did not produce a supported answer.")
    for item in sorted(evidence, key=lambda hit: hit["id"] not in cited):
        used = item["id"] in cited
        label = f"{item['section']} · {item['id']}"
        if used:
            label += " · Cited"
        with right.expander(label, expanded=used):
            score_label = item.get("score_label", "Cosine similarity")
            st.caption(f"{item.get('paper_id', paper_id)} · {score_label} {item['similarity']:.3f}")
            st.text(item["text"])
