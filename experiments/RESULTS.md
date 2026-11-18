# Paired benchmark reanalysis

The two published benchmarks use different question sets and evidence definitions. The comparisons below pair identical questions **within** each benchmark. Reproduce this analysis with `python experiments/compare_saved.py`. It uses archived per-question logs and requires only Python.

## Effect of reranking in the Qwen benchmark

The paired comparison holds chunking, retrieval scope, questions, and the final Top-5 cutoff fixed. Reranking takes 24 retrieval candidates and returns five. Hit@5 uses the upstream Qwen definition, in which a top-five chunk must contain a complete normalized evidence string. The hit denominators are 40 answerable questions per pair. The 25 unanswerable questions are used separately to assess refusal.

| Chunking | Scope | Hits before | Hits after | Gained | Lost | MRR before → after | Correct refusals |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Structured sections | Selected paper | 29/40 | 35/40 | 7 | 1 | 0.436 → 0.512 | 21/25 → 23/25 |
| Structured sections | Global | 10/40 | 13/40 | 3 | 0 | 0.160 → 0.209 | 21/25 → 21/25 |
| Sliding window | Selected paper | 29/40 | 27/40 | 0 | 2 | 0.337 → 0.403 | 21/25 → 21/25 |
| Sliding window | Global | 12/40 | 12/40 | 2 | 2 | 0.102 → 0.163 | 21/25 → 19/25 |

Structured sections with selected-paper retrieval gained six hits overall. Reranking improved mean reciprocal rank in all four archived comparisons. For global sliding windows, the equal hit totals conceal two recovered questions and two regressions. Upstream refusal thresholds were calibrated separately for each configuration, so the refusal changes describe complete configurations rather than an isolated ranking effect.

## Effect of chunking in the PaperScope benchmark

Both strategies hit evidence on 14/20 answerable questions using the original paragraph-coverage rule. Paragraph-aware chunking recovered two cases and lost two others. Strict answer-span matches went from 6/20 to 8/20 with three gains and one loss. The original validator withheld one paragraph-aware answer due to a citation error.

## Shared-set experiment

`shared_cases.json` fixes the same 20 extractive QASPER questions and paper IDs for both retrieval backends. `run_shared.py` uses the selected-paper scope, Top-5, and the same normalized complete-evidence-string rule. It exports ordered passages for manual answer and citation-support review when run with `--answer`. The proposed BGE and Qwen configurations use their own chunkers. A result from these settings compares complete retrieval configurations.

A live BGE-small retrieval run used the same 20 papers and questions, selected-paper retrieval, and the same Top-5 scoring rule for both chunkers.

| BGE-small chunking | Hit@5 |
| --- | ---: |
| Fixed size | 9/20 (45%) |
| Paragraph aware | 14/20 (70%) |

Paragraph-aware chunking recovered six questions and lost one. The [per-question hit records](shared_bge_retrieval.json) identify the paired outcomes. DeepSeek generation, manual answer review, and Qwen retrieval on the shared set are pending.

- [PaperScope original benchmark](https://github.com/Kevin999-sudo/paperscope-cs5481/tree/7f02989d3410532eeed12329ea82ef2369b24a71/results)
- [Qwen original benchmark](https://github.com/a5ynchr0ny/CS5481-project-rag/tree/e0493416a87dc60f7e5762a548ca999fe565c60c/results)
- [Qwen comparison notes](https://github.com/a5ynchr0ny/CS5481-project-rag/blob/e0493416a87dc60f7e5762a548ca999fe565c60c/COMPARE.md)
