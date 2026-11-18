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

| Model and index | Fixed-size Hit@5 | Paragraph-aware Hit@5 |
| --- | ---: | ---: |
| BGE-small, Qdrant in memory | 9/20 (45%) | 14/20 (70%) |
| BGE-base, persistent Qdrant | 8/20 (40%) | 11/20 (55%) |

Paragraph-aware chunking recovered six questions and lost one. The [per-question hit records](shared_bge_retrieval.json) identify the paired outcomes. Both BGE-small configurations then generated 20 answers. Exact quote checks resolved 54/54 fixed-size citations and 57/57 paragraph-aware citations against the retrieved passages. Answer correctness and citation support still await human review.

BGE-base retrieved on the identical question set under the same rule. The
fixed-size and paragraph-aware runs used the same 768-dimensional disk index
across separate processes. See the [BGE-base paired records](shared_base_retrieval.json).
On this sample, BGE-small with paragraph-aware chunking had the highest Hit@5.

## Structured Checked

The structured Checked workflow returns a row for each material condition with the
assertion, evidence finding, verdict, and cited excerpt where applicable.
PaperScope validates the source and quotation, then computes the overall
verdict. The [claim ablation entry point](../README.md#direct-and-checked-statement-comparison)
runs the four controlled configurations. The live run `eac2c692e5b98701` held
BGE-small, DeepSeek Flash, the 40 statement labels and Top-5 constant. Each
within-chunker Direct/Checked pair received identical evidence. All 160 API
calls produced results, and all 80 Checked results passed citation validation
and included structured conditions.

| Chunking | Method | Correct | Macro-F1 | Gold-supported retained | False acceptance | Median latency | Total tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed size | Direct | 39/40 | 0.972 | 19/20 | 0/20 | 1.08 s | 54,274 |
| Fixed size | Checked | 34/40 | 0.832 | 16/20 | 0/20 | 1.46 s | 61,017 |
| Paragraph aware | Direct | 39/40 | 0.972 | 19/20 | 0/20 | 1.00 s | 48,879 |
| Paragraph aware | Checked | 35/40 | 0.872 | 15/20 | 0/20 | 1.56 s | 55,393 |

Structured Checked exposed which conditions triggered the verdict. In this
40-statement set it lost five correct classifications for fixed chunks and
four for paragraph-aware chunks, with zero gains against Direct. One example
is `claim-02-a`, where paragraph-aware Checked verified the sentence count
and manual annotation but marked the dataset's test role insufficient. The
extra condition changed a gold-supported statement to insufficient.
This comparison measures verdict agreement with the frozen labels. Human
evaluation of the condition statements and citation support remains separate.
See [compact per-case results](claim_ablation_small.json). Earlier prompt-based
Checked verdicts remain in the original archive.

- [PaperScope original benchmark](https://github.com/Kevin999-sudo/paperscope-cs5481/tree/7f02989d3410532eeed12329ea82ef2369b24a71/results)
- [Qwen original benchmark](https://github.com/a5ynchr0ny/CS5481-project-rag/tree/e0493416a87dc60f7e5762a548ca999fe565c60c/results)
- [Qwen comparison notes](https://github.com/a5ynchr0ny/CS5481-project-rag/blob/e0493416a87dc60f7e5762a548ca999fe565c60c/COMPARE.md)
