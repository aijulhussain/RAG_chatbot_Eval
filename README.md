# RAG Eval Project

A Retrieval-Augmented Generation (RAG) teaching-assistant pipeline, built on a corpus of
LLM-evaluation lecture transcripts, together with a full evaluation harness that scores the
pipeline across quality, safety, and operational dimensions.

The bot answers student questions about LLM evals using **only** its retrieved course
content, and the eval suite checks every leg of that promise: did retrieval find the right
chunks, is the answer faithful to them, does the bot stay in scope, does it leak anything it
shouldn't, and is it fast/cheap/reliable enough to run in production.

## How it works

```
query ──▶ retriever (Chroma + OpenAI embeddings, top fetch_k)
            │
            ▼
        cross-encoder reranker (scores query+chunk pairs, keeps top_k)
            │
            ▼
        generator (gpt-4o-mini, faithfulness-first prompt)
            │
            ▼
        answer (+ query, + context returned for eval)
```

- **Corpus**: `.vtt` lecture transcripts in [data/](data/) (8 "LLM Evals" sessions), cleaned
  and chunked with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=150).
- **Store**: persisted to [chroma_store/](chroma_store/) on first run; later runs load it
  from disk instead of re-embedding.
- **Retriever** ([src/retriever.py](src/retriever.py)): similarity search over the Chroma
  store using `text-embedding-3-small`.
- **Reranker** ([src/reranker.py](src/reranker.py)): over-fetches `fetch_k` candidates, then
  re-scores each `(query, chunk)` pair with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and keeps the best `top_k`.
- **Generator** ([src/generator.py](src/generator.py)): `gpt-4o-mini` behind a long,
  deliberately defensive prompt — answer only from context, abstain when unsure, refuse to
  leak its own instructions or the raw course corpus, stay non-toxic, don't go out of scope.
  A `generate_stream` twin exists for time-to-first-token measurement.
- **Pipeline** ([src/rag_pipeline.py](src/rag_pipeline.py)): wires the three stages together
  and returns `{"query", "context", "answer"}` so the eval harness can score any leg of the
  triad independently.

## Setup

Requires Python >=3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/)
(`pyproject.toml` / `uv.lock`), with a `requirements.txt` as a plain-pip fallback.

```bash
uv sync
# or: pip install -r requirements.txt
```

Create a `.env` file with your API keys (the generator uses OpenAI directly; OpenRouter is
wired in as an available alternative chat model):

```
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

First run builds the Chroma store from `data/*.vtt` (embeds everything, takes a moment);
later runs reuse [chroma_store/](chroma_store/). Delete that directory to force a rebuild
after changing the corpus or chunking parameters.

## Running the pipeline

```bash
python -m src.rag_pipeline
```

Runs one smoke-test query end to end and prints the query, answer, and retrieved chunks.

## Evaluations

All evals use [deepeval](https://docs.confident-ai.com/) and read golden/test data from
[golden_data/](golden_data/). Most use an LLM judge (`gpt-4.1-mini` or `gpt-4o-mini`) scored
against a `threshold`, and are run directly as scripts:

```bash
python -m eval.<eval_name>
```

### Component-level

| Script | Checks | Metrics |
|---|---|---|
| [eval_retriever.py](eval/eval_retriever.py) | Plain similarity-search retriever in isolation | `ContextualRecallMetric`, `ContextualPrecisionMetric` |
| [eval_retriever_reranker.py](eval/eval_retriever_reranker.py) | Retriever + cross-encoder reranker, same metrics — compare against the row above to see what reranking buys you | `ContextualRecallMetric`, `ContextualPrecisionMetric` |
| [eval_generator.py](eval/eval_generator.py) | Generator fed **golden** (known-good) context, isolating generator faults from retrieval faults | `FaithfulnessMetric`, `AnswerRelevancyMetric` |

### Full-pipeline (triad + application quality)

| Script | Checks | Metrics |
|---|---|---|
| [eval_rag_pipeline.py](eval/eval_rag_pipeline.py) | Retrieve → rerank → generate, live context and answer | `ContextualRelevancyMetric`, `FaithfulnessMetric`, `AnswerRelevancyMetric` |
| [eval_application.py](eval/eval_application.py) | Reference-based correctness/completeness + reference-free teaching style, via custom `GEval` rubrics | `GEval` (Correctness, Completeness, Style) |

### Safety

| Script | Checks | Metrics |
|---|---|---|
| [eval_toxicity.py](eval/eval_toxicity.py) | Adversarial/abusive inputs don't produce toxic output | `ToxicityMetric` |
| [eval_leakage.py](eval/eval_leakage.py) | Three leakage surfaces, each its own golden subset: hidden system-prompt leakage, verbatim course-corpus leakage, PII leakage | `GEval` (Prompt Leakage, Course Content Leakage), `PIILeakageMetric` |
| [eval_scope.py](eval/eval_scope.py) | Bot stays a course TA — answers in-scope questions, declines unrelated/jailbreak requests, splits mixed requests | `GEval` (Scope Adherence) |

### Operational (deterministic — no LLM judge, no golden set)

| Script | Checks | Approach |
|---|---|---|
| [eval_latency.py](eval/eval_latency.py) | End-to-end and time-to-first-token latency against SLO budgets (p95 ≤ 3000ms / ≤ 1200ms TTFT) | Warm up, sample N repeats per question, report percentiles |
| [eval_cost.py](eval/eval_cost.py) | Per-query token usage → USD/INR cost, with cached-input-token discount modeled, projected to daily/monthly spend | Real token counts via `usage_metadata`, multiplied by per-1M-token pricing |
| [eval_reliability.py](eval/eval_reliability.py) | Success rate, error rate, retry rate under repeated calls | Retry wrapper with exponential backoff around `pipeline.invoke` |

## Golden / test data

Hand-authored and synthesizer-drafted datasets live in [golden_data/](golden_data/):

- `retriever_goldens.json`, `retriever_deepeval_goldens.json` — query + ideal answer, for
  retrieval recall/precision.
- `faithfulness_dataset.json` — query + known-good context, for generator-only faithfulness.
- `correctness_dataset.json` — query + ideal answer, for full-pipeline correctness/completeness/style.
- `toxicity_dataset.json` — adversarial/abusive inputs.
- `leakage_dataset.json` — `prompt` / `course_content` / `pii` subtypes, each with an
  `expected_action`.
- `scope_dataset.json` — benign, jailbreak, and mixed requests, each with an
  `expected_action` (`ANSWER` / `DECLINE` / `PARTIAL`) and `success_criteria`.
- `golden_generator.py` — draws a random sample of corpus chunks and uses deepeval's
  `Synthesizer` to draft new retriever goldens. Output is explicitly marked as a draft —
  review every row before trusting it (grounding, leading questions, correct `source`).

## Project layout

```
src/            retriever, reranker, generator, and the assembled pipeline
eval/           one script per evaluation (component, full-pipeline, safety, operational)
golden_data/    golden/test datasets + the synthetic-golden generator
data/           raw .vtt lecture transcripts (the source corpus)
chroma_store/   persisted vector store (generated, not source-controlled content)
```
