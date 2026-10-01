"""
Operational eval: RELIABILITY

Reliability measures whether the RAG application can successfully
serve requests without failing.

We measure:
    - success rate
    - error rate
    - retry rate

Retries are important because a system may eventually succeed while
still being flaky on the first attempt.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import time
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline

load_dotenv()


# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5

MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


# ============================================================
# 3. RELIABILITY TRACKER
# ============================================================
class Reliability:

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0


# ============================================================
# 4. RETRY WRAPPER
# ============================================================
def call_with_retries(fn, reliability):

    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):

        try:
            result = fn()

            reliability.successes += 1

            return result

        except Exception as e:

            if attempt < MAX_RETRIES:

                reliability.retries += 1

                # exponential backoff
                time.sleep(
                    BACKOFF_BASE_S * (2 ** attempt)
                )

            else:

                reliability.failures += 1

                print(
                    f"FAILED after {MAX_RETRIES} retries: {e}"
                )

                return None


# ============================================================
# 5. BENCHMARK
# ============================================================
def benchmark(pipeline):

    reliability = Reliability()

    print("Measuring reliability...")

    for question in QUESTIONS:

        for _ in range(REPEATS):

            call_with_retries(
                lambda: pipeline.invoke(question),
                reliability
            )

    return reliability


# ============================================================
# 6. REPORT
# ============================================================
def report(rel):

    success_rate = (
        100 * rel.successes / rel.calls
        if rel.calls else 0
    )

    error_rate = (
        100 * rel.failures / rel.calls
        if rel.calls else 0
    )

    retry_rate = (
        100 * rel.retries / rel.calls
        if rel.calls else 0
    )

    print("\n" + "=" * 60)
    print("RELIABILITY")
    print("=" * 60)

    print(f"total requests : {rel.calls}")
    print(f"successful     : {rel.successes}")
    print(f"failed         : {rel.failures}")

    print("-" * 60)

    print(f"success rate   : {success_rate:.2f}%")
    print(f"error rate     : {error_rate:.2f}%")
    print(f"retry rate     : {retry_rate:.2f}%")

    print("=" * 60)


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():

    pipeline = RagPipeline()

    reliability = benchmark(pipeline)

    report(reliability)


if __name__ == "__main__":
    main()