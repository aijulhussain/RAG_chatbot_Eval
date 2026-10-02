# eval/mlflow_logger.py
"""
Log a regression_suite.py snapshot to MLflow.

One snapshot -> one MLflow run:
  - metadata (git_sha, prompt_hash, label, timings) -> params
  - metrics (the flat dotted metric ids)             -> metrics (bool SLO/budget
                                                         flags cast to 0.0/1.0,
                                                         since MLflow metrics must
                                                         be numeric)
  - the raw snapshot JSON                            -> artifact

This is a logging/visualization layer only -- it does NOT replace
metric_registry.py + compare.py. The gate/guardrail/tolerance PASS/REVIEW/FAIL
decision still runs exactly as before; MLflow just gives you a searchable
history of every run instead of two JSON files on disk (baseline/candidate).

Uses the local ./mlruns store by default (no server needed). Point
MLFLOW_TRACKING_URI at a remote server if you want a shared one.

    python -m eval.regression_suite --label "..."      # writes baselines/candidate.json
    python -m eval.mlflow_logger baselines/candidate.json
    python -m eval.regression_suite --label "..." --mlflow   # logs automatically
"""

import json
import os
import re
import sys
import tempfile

import mlflow

EXPERIMENT_NAME = "rag-eval-regression"

# MLflow metric/param names may only contain alphanumerics, '_', '-', '.', ' ',
# and '/'. Our ids can carry GEval's own metric names verbatim (e.g.
# "Correctness [GEval]" -> "application.correctness_[geval].avg_score"), so
# square brackets (and anything else disallowed) get swapped for '_'.
_DISALLOWED_NAME_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")


def _sanitize_name(name):
    return _DISALLOWED_NAME_CHARS.sub("_", name)


def _as_metric(value):
    """MLflow metrics must be numeric; cast bool SLO/budget flags to 0.0/1.0."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def log_snapshot(snapshot, run_name=None):
    """Log one snapshot dict (as produced by regression_suite.run_suite) as an
    MLflow run. Returns the run id."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    meta = snapshot.get("metadata", {})
    metrics = snapshot.get("metrics", {})

    with mlflow.start_run(run_name=run_name or meta.get("label") or None) as run:
        mlflow.log_params({
            "git_sha": meta.get("git_sha", "unknown"),
            "prompt_hash": meta.get("prompt_hash", "unknown"),
            "label": meta.get("label", ""),
            "full": meta.get("full", False),
            "n_metrics": meta.get("n_metrics", len(metrics)),
        })
        mlflow.log_metric("suite_seconds", float(meta.get("suite_seconds", 0.0)))
        mlflow.log_metrics({_sanitize_name(mid): _as_metric(v) for mid, v in metrics.items()})

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2, sort_keys=True)
            mlflow.log_artifact(path)

        print(f"[mlflow] logged run {run.info.run_id} -> experiment '{EXPERIMENT_NAME}'")
        return run.info.run_id


def main():
    if len(sys.argv) < 2:
        print("usage: python -m eval.mlflow_logger <snapshot.json> [run_name]")
        sys.exit(1)

    path = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else None

    with open(path) as f:
        snapshot = json.load(f)

    log_snapshot(snapshot, run_name=run_name)


if __name__ == "__main__":
    main()
