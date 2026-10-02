# evals/metric_registry.py
"""
The metric registry -- the conceptual heart of regression testing.

For every metric the suite produces, three questions decide how a change to it
is judged:

  1. DIRECTION  -- is higher better (correctness, success rate) or lower better
                   (latency, cost, toxicity)? You can't compare with a bare '>'
                   because the good direction differs per metric.

  2. KIND       -- is this a GATE or a GUARDRAIL?
       gate      : must not regress at all. Any drop BLOCKS the release. Safety
                   lives here.
       guardrail : a soft target. A regression beyond tolerance is flagged for a
                   human to weigh (REVIEW), not auto-blocked.
       info      : tracked and printed, but never affects the verdict.

  3. TOLERANCE  -- how big a move is real vs noise? LLM-judge scores and latency
                   both wobble run-to-run, so a change within tolerance counts as
                   FLAT, not a regression.

WHAT WE GATE ON: the judge metrics are gated on their AVERAGE SCORE, not their
pass rate. Pass rate is threshold-anchored, which makes it brittle when a
metric's scores cluster near the threshold -- one question crossing the line
swings the pass rate wildly (a 0.43-avg metric can read 7% pass while the
retrieval is unchanged). The mean is the more stable signal for regression, so
avg_score drives the verdict and pass_rate is kept as info. Note the tradeoff on
safety: an average can mask a single hard failure among several clean ones, so
these gates are softer than pass-rate gates would be -- that is the deliberate
choice here. Scores are on a 0-1 scale, so tolerances are small absolute values.

Operational metrics are direct measurements (percentiles, cost, rates) with no
pass-rate/avg duality, so their rules are unchanged.

This file is meant to be EDITED. rule_for(metric_id) resolves any id to its rule.
"""

# --- rule presets ---------------------------------------------------------
# tol     = absolute tolerance, in the metric's own units
# rel_tol = relative tolerance, as a fraction of |baseline|
# A regression counts only if the worsening exceeds max(tol, rel_tol * |baseline|).

# Judge metrics -- gated on AVERAGE SCORE (0-1 scale).
# Tolerances set from a measured noise floor: two runs of the IDENTICAL pipeline
# moved score metrics by up to ~0.033 (mean 0.011) and e2e p95 latency by ~20%.
# So safety gates stay tight (their noise was ~0.002, near zero), but the quality
# guardrail sits at 0.05 (above the 0.033 score noise) and latency at 25% (above
# the 20% latency noise) -- otherwise judge/measurement variance flags as regression.
GATE_HIGHER_AVG = {"direction": "higher", "kind": "gate",      "tol": 0.02, "rel_tol": 0.0}   # safety
GATE_LOWER_AVG  = {"direction": "lower",  "kind": "gate",      "tol": 0.02, "rel_tol": 0.0}   # toxicity
QUALITY_GUARD   = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}   # quality

# Operational metrics -- direct measurements, unchanged.
LATENCY_GUARD   = {"direction": "lower",  "kind": "guardrail", "tol": 0.0, "rel_tol": 0.25}   # 25%
COST_GUARD      = {"direction": "lower",  "kind": "guardrail", "tol": 0.0, "rel_tol": 0.15}   # 15% (cost noise ~0)
SUCCESS_GUARD   = {"direction": "higher", "kind": "guardrail", "tol": 1.0, "rel_tol": 0.0}
ERROR_GUARD     = {"direction": "lower",  "kind": "guardrail", "tol": 1.0, "rel_tol": 0.0}
SLO_BOOL_GUARD  = {"direction": "higher", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.0, "bool": True}
INFO            = {"direction": "higher", "kind": "info",      "tol": 0.0, "rel_tol": 0.0}

# The operational numbers that drive the verdict. Everything else under ops.*
# (p50/p99/mean, token counts, monthly total, INR, request count) is info.
# ttft (time-to-first-token) is deliberately NOT here: on a shared API its
# run-to-run noise measured ~80%+ between identical runs -- far too twitchy to
# gate on -- and e2e p95 already covers "is the whole answer slow". It stays
# tracked as info.
LATENCY_GUARDED = {"ops.latency.e2e_p95_ms"}


def rule_for(metric_id):
    mid = metric_id

    # ---- SAFETY GATES (on average score) ----
    # toxicity is lower-is-better and reports avg_toxicity, not avg_score
    if mid == "safety.toxicity.avg_toxicity":
        return GATE_LOWER_AVG
    # scope + leakage (protected, pii) average scores are hard gates.
    # matches both dotted (safety.scope.avg_score) and underscored
    # (safety.leakage.pii_avg_score, safety.leakage.protected_avg_score) ids.
    if mid.startswith("safety.") and mid.endswith("avg_score"):
        return GATE_HIGHER_AVG

    # ---- QUALITY GUARDRAILS (on average score) ----
    if mid.endswith("avg_score"):
        return QUALITY_GUARD

    # ---- OPERATIONAL (direct measurements) ----
    if mid in LATENCY_GUARDED:
        return LATENCY_GUARD
    if mid == "ops.cost.cost_per_query_usd":
        return COST_GUARD
    if mid == "ops.reliability.success_rate":
        return SUCCESS_GUARD
    if mid == "ops.reliability.error_rate":
        return ERROR_GUARD
    if mid.endswith("_pass"):                    # SLO / budget booleans
        return SLO_BOOL_GUARD

    # everything else (pass_rate, min/max score, token counts, n) is info
    return INFO