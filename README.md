# PaperScope Fusion

PaperScope is a CS5481 Topic 5 system for research-paper question answering and statement checking with verifiable citations. The Streamlit app combines the original PaperScope answer and citation workflow with an optional Qwen retrieval backend. The default BGE-small and Qdrant path supports CPU use and local extractive QA.

## What is integrated

- **PaperScope** supplies the interface, BGE-small retrieval, DeepSeek generation, direct and checked verdicts, and validation of quoted text.
- **Qwen RAG** supplies structured and sliding-window chunkers, Ollama embeddings, Chroma retrieval, and optional Qwen3 reranking. Its passages enter the same PaperScope answer and citation workflow.
- **Evaluation** shows a paired reanalysis of both archived benchmarks. A separate runner fixes the same 20 QASPER questions and scoring rule for a live comparison when both backends are available.

## Reproduce the published analysis

~~~sh
git clone https://github.com/wuyu200501/paperscope-fusion.git
cd paperscope-fusion
python experiments/compare_saved.py
python build_replay.py
~~~

These commands use only the Python standard library and the included benchmark
records. The second script rebuilds `PaperScope-Replay.html` with 64 archived
cases and 208 outputs. The archived Qwen comparison pairs 40 answerable
questions per saved configuration. The archived PaperScope comparison pairs
20 answerable questions across chunkers. See the
[results and limitations](experiments/RESULTS.md) for their scoring definitions.

## Run PaperScope

Use Python 3.12 in a virtual environment. Preparing assets downloads the QASPER archive and the local BGE and MiniLM weights.

~~~sh
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
python prepare_assets.py
python run_deepseek.py --serve --offline
~~~

For Windows PowerShell, create the environment with `python -m venv .venv`
and use `& .\.venv\Scripts\python.exe` in place of `python` in the commands above.

Open [the local app](http://127.0.0.1:8517). For DeepSeek answers and statement checking run `python run_deepseek.py --serve` and enter a key at the hidden terminal prompt. You may also set `PAPERSCOPE_API_KEY` in your local session.

Offline smoke test on the **pioNER** paper: select **BGE-small** and
**Paragraph aware**, then ask `What is the source of the news sentences?`.
The expected answer is `ilur.am`. The **Evaluation** page displays the
published structured-checking results without making API requests.

For authentication troubleshooting, run
`python run_deepseek.py --check-key --serve` in an interactive terminal. The
launcher prompts for a fresh key, checks it with DeepSeek's models endpoint,
and starts the app with the same key after the check passes.
If pasting into the hidden prompt adds unexpected characters, set
`PAPERSCOPE_API_KEY` in the current shell and add `--use-env-key` to use that
value for the check.

On Windows, PaperScope loads the system Visual C++ runtime before PyTorch to
avoid a `WinError 1114` from older runtime DLLs bundled with some Conda
installations. If the system runtime needs updating, use the [Microsoft
Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### Enable Qwen retrieval

Install the extra dependency, start Ollama, and install the embedding model. An existing Chroma index uses `PAPERSCOPE_QWEN_DB` to point at its directory. To create a fresh index after preparing the QASPER archive:

~~~sh
python -m pip install -r requirements-qwen.txt
ollama pull qwen3-embedding:4b-fp16
python -m qwen_backend.prepare_corpus --data data/qasper-train-dev-v0.3.tgz --out data/chunks
python -m qwen_backend.build_index --chunks data/chunks --db data/chroma
python run_deepseek.py --serve
~~~

Choose **Qwen** in the retrieval settings. The optional rerank checkbox expects a running Qwen3-Reranker-0.6B server, configurable through `RERANK_URL`. See the [upstream reranker setup](https://github.com/a5ynchr0ny/CS5481-project-rag/blob/e0493416a87dc60f7e5762a548ca999fe565c60c/README.md#optional-cross-encoder-reranking-without-torch). Qwen retrieves passages in the selected paper in the app. The archived global-search results remain available on the Evaluation page.

The shared-set runner also accepts Qwen once the local index is available.
Add `--rerank` only when the reranker service is running.

~~~sh
python -m experiments.run_shared --backend qwen --strategy A --output data/shared-qwen.json
python -m experiments.run_shared --backend qwen --strategy A --rerank --output data/shared-qwen-rerank.json
~~~

## Run the shared experiment

~~~sh
python -m experiments.run_shared --backend bge --strategy fixed --output data/shared-bge-fixed-retrieval.json
python -m experiments.run_shared --backend bge --strategy section --output data/shared-bge-section-retrieval.json
~~~

The published BGE-small Hit@5 values for these two commands are **9/20**
and **14/20**. This retrieval comparison runs locally.

Add `--answer` and set `PAPERSCOPE_API_KEY` locally to generate answers in addition to retrieval output. For human review, check answer correctness against the reference first. Then check whether the displayed cited passages support the answer and record the two judgments separately. A cross-backend score belongs in the joint report after both runs and reviews have been completed.

### BGE-base and persistent Qdrant

~~~sh
python prepare_assets.py --embedding base
python -m experiments.run_shared --backend bge --embedding base --strategy fixed --output data/shared-base-fixed.json
python -m experiments.run_shared --backend bge --embedding base --strategy section --output data/shared-base-section.json
~~~

Select **BGE-base** in the app or pass `--embedding base` to the benchmark.
BGE-base reuses a disk-backed Qdrant index under `data/qdrant`. The model
revision, source corpus, and chunking settings determine the index directory.
The BGE-small baseline uses the original in-memory configuration by default.
Stop other local PaperScope processes if a Qdrant directory is locked.

### Direct and Checked statement comparison

In **Check a statement**, choose **Checked** for a condition table or
**Direct** for the original verdict. Each structured row shows the submitted
condition, the evidence finding, and a verifiable quotation when available.
The overall verdict follows conflict, insufficient evidence, then support.
On the **pioNER** paper, compare `The manually annotated test dataset contains
2,566 sentences.` with the same claim using `5,266 sentences`. For missing
evidence, try `Both corpus creation procedures would achieve equally accurate
results for Georgian.`.

~~~sh
python run_deepseek.py --check-key --claim-ablation --embedding small
~~~

The command prompts for a DeepSeek key and evaluates 40 existing statements
across the four fixed/paragraph × Direct/Checked configurations. Each pair
uses identical retrieved passages and the same output token cap. The
checkpoint reuses the paired evidence when retrying a failed API call.
See `data/claim-ablation-small/metrics.json` for accuracy, macro-F1, paired
success, false acceptance, latency, and token totals. The older archived
experiment used prompt-based Checked; the new run writes to a separate folder.
The same command with `--embedding base` runs an independent BGE-base
comparison. The [published BGE-small run](experiments/claim_ablation_small.json)
contains the four configuration summaries and compact per-case records.
Fresh DeepSeek calls can vary, so report their own run fingerprint and scores.

The original PaperScope archive stays under `results/`. A full fresh benchmark
launched with `python run_deepseek.py --check-key --evaluate` writes to
`data/live-benchmark-small/`. Use
`python build_replay.py --from-run data/live-benchmark-small --output PaperScope-Replay-latest.html`
to build a standalone replay from that completed run.

## Source attribution

- [Kevin999-sudo/paperscope-cs5481](https://github.com/Kevin999-sudo/paperscope-cs5481/tree/7f02989d3410532eeed12329ea82ef2369b24a71) for the original app, benchmark and statements.
- [a5ynchr0ny/CS5481-project-rag](https://github.com/a5ynchr0ny/CS5481-project-rag/tree/e0493416a87dc60f7e5762a548ca999fe565c60c) for the Qwen retrieval, reranking, archived logs and comparison notes.
- [QASPER](https://aclanthology.org/2021.naacl-main.365/) supplies papers and human-labeled QA evidence. The [dataset](https://huggingface.co/datasets/allenai/qasper) is available under CC BY 4.0.

Kevin's statement definitions and four original result files retain their
upstream Git blob hashes. The Qwen evaluation snapshot retains the fields
used in this project from 520 records across eight upstream configurations.
The Qwen modules in `qwen_backend/` carry the launcher, format, and import
adaptations used by this repository.
