# PaperScope Fusion

PaperScope is a CS5481 Topic 5 system for research-paper question answering and statement checking with verifiable citations. The Streamlit app combines the original PaperScope answer and citation workflow with an optional Qwen retrieval backend. The default BGE-small and Qdrant path supports CPU use and local extractive QA.

## What is integrated

- **PaperScope** supplies the interface, BGE-small retrieval, DeepSeek generation, direct and checked verdicts, and validation of quoted text.
- **Qwen RAG** supplies structured and sliding-window chunkers, Ollama embeddings, Chroma retrieval, and optional Qwen3 reranking. Its passages enter the same PaperScope answer and citation workflow.
- **Evaluation** shows a paired reanalysis of both archived benchmarks. A separate runner fixes the same 20 QASPER questions and scoring rule for a live comparison when both backends are available.

## Reproduce the published analysis

~~~sh
python experiments/compare_saved.py
~~~

This command uses the Python standard library and the included public benchmark records. The Qwen part compares the same 40 answerable questions with and without reranking for each saved configuration. The PaperScope part pairs its original 20 answerable questions across chunkers. The two benchmarks use different question sets and hit definitions. See the [results and limitations](experiments/RESULTS.md).

## Run PaperScope

Use Python 3.12 in a virtual environment. Preparing assets downloads the QASPER archive and the local BGE and MiniLM weights.

~~~sh
python -m pip install -r requirements.txt
python prepare_assets.py
python run_deepseek.py --serve --offline
~~~

Open [the local app](http://127.0.0.1:8517). For DeepSeek answers and statement checking run `python run_deepseek.py --serve` and enter a key at the hidden terminal prompt. You may also set `PAPERSCOPE_API_KEY` in your local session.

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

## Run the shared experiment

~~~sh
python -m experiments.run_shared --backend bge --strategy section --output data/shared-bge.json
python -m experiments.run_shared --backend qwen --strategy A --output data/shared-qwen.json
python -m experiments.run_shared --backend qwen --strategy A --rerank --output data/shared-qwen-rerank.json
~~~

Add `--answer` and set `PAPERSCOPE_API_KEY` locally to generate answers in addition to retrieval output. For human review, check answer correctness against the reference first. Then check whether the displayed cited passages support the answer and record the two judgments separately. A cross-backend score belongs in the joint report after both runs and reviews have been completed.

To generate an optional offline replay of the original PaperScope records, run `python build_replay.py`. For a completed new PaperScope run, use `python build_replay.py --from-run results --output PaperScope-Replay-latest.html`.

## Source attribution

- [Kevin999-sudo/paperscope-cs5481](https://github.com/Kevin999-sudo/paperscope-cs5481/tree/7f02989d3410532eeed12329ea82ef2369b24a71) for the original app, benchmark and statements.
- [a5ynchr0ny/CS5481-project-rag](https://github.com/a5ynchr0ny/CS5481-project-rag/tree/e0493416a87dc60f7e5762a548ca999fe565c60c) for the Qwen retrieval, reranking, archived logs and comparison notes.
- [QASPER](https://aclanthology.org/2021.naacl-main.365/) supplies papers and human-labeled QA evidence. The [dataset](https://huggingface.co/datasets/allenai/qasper) is available under CC BY 4.0.
