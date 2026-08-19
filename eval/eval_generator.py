"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault — the context was already correct.

    python -m evals.eval_generator
"""

import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate   # your generator: generate(query, context) -> answer

load_dotenv()

GOLDEN_PATH = "golden_data/faithfulness_dataset.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


# 1. LOAD the faithfulness golden set (query + ideal_context)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
test_cases = []
for g in goldens:
    context = g["ideal_context"]              # known-good context (list of chunk strings)
    answer = generate(g["query"], context)    # RUN the generator -> actual_output

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             # the generated answer we're judging
            retrieval_context=context,        # faithfulness checks the answer against THIS
            # no expected_output — faithfulness never reads it
        )
    )


# 3. THE METRIC — decomposes actual_output into claims, attributes each to context
metrics = [FaithfulnessMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=True,   # prints WHY each score — shows which claims were unsupported
),
AnswerRelevancyMetric(
    threshold=THRESHOLD, 
    model=JUDGE_MODEL, 
    include_reason=True)
]


# 4. EVALUATE — runs the metric on every case, prints a report
evaluate(test_cases=test_cases, metrics=metrics)