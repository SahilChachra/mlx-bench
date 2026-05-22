"""
Download and cache all benchmark datasets.
Run once before any evaluation.
"""

import json
from pathlib import Path
from datasets import load_dataset

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} samples → {path}")


def setup_gsm8k(n=30):
    print("Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    samples = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    save_jsonl(samples, DATASETS_DIR / "gsm8k.jsonl")


def setup_humaneval(n=30):
    print("Downloading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split=f"test[:{n}]")
    samples = [
        {
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "canonical_solution": r["canonical_solution"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        }
        for r in ds
    ]
    save_jsonl(samples, DATASETS_DIR / "humaneval.jsonl")


def setup_mmlu(n=50):
    print("Downloading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
    samples = [
        {
            "question": r["question"],
            "choices": r["choices"],
            "answer": int(r["answer"]),
            "subject": r["subject"],
        }
        for r in ds
    ]
    save_jsonl(samples, DATASETS_DIR / "mmlu.jsonl")


def setup_long_context():
    print("Writing long-context prompts...")
    prompts = [
        {
            "id": "lc_01",
            "prompt": "Write a detailed technical explanation of how transformer attention mechanisms work, covering self-attention, multi-head attention, positional encodings, and their computational complexity. Include worked examples.",
            "expected_min_tokens": 400,
        },
        {
            "id": "lc_02",
            "prompt": "Explain the entire history of machine learning from the perceptron in the 1950s to modern large language models. Cover key milestones, researchers, architectural innovations, and paradigm shifts.",
            "expected_min_tokens": 500,
        },
        {
            "id": "lc_03",
            "prompt": "Write a complete Python implementation of a binary search tree with insert, delete, search, in-order traversal, and balancing. Include docstrings and unit tests.",
            "expected_min_tokens": 400,
        },
        {
            "id": "lc_04",
            "prompt": "Describe in detail how TCP/IP works from the physical layer to the application layer. Include handshake processes, congestion control, error handling, and differences from UDP.",
            "expected_min_tokens": 400,
        },
        {
            "id": "lc_05",
            "prompt": "Write a detailed essay on the economic implications of artificial intelligence on the global labor market over the next 20 years, covering automation risk, new job creation, policy responses, and regional disparities.",
            "expected_min_tokens": 500,
        },
        {
            "id": "lc_06",
            "prompt": "Provide a comprehensive overview of quantization methods for neural networks: post-training quantization, quantization-aware training, GPTQ, AWQ, GGUF, and MLX formats. Compare quality/speed tradeoffs.",
            "expected_min_tokens": 400,
        },
        {
            "id": "lc_07",
            "prompt": "Explain how operating system kernels manage memory: virtual memory, paging, segmentation, TLB, page faults, and memory protection. Include diagrams in ASCII.",
            "expected_min_tokens": 400,
        },
        {
            "id": "lc_08",
            "prompt": "Write a complete guide to building a REST API in Python with FastAPI including authentication, rate limiting, database integration, error handling, and deployment. Include full code examples.",
            "expected_min_tokens": 500,
        },
        {
            "id": "lc_09",
            "prompt": "Summarize and analyze the key ideas in 'Attention Is All You Need', 'BERT', 'GPT-3', 'InstructGPT', and 'Constitutional AI' papers. What does each paper contribute and how do they build on each other?",
            "expected_min_tokens": 400,
        },
        {
            "id": "lc_10",
            "prompt": "Describe the full lifecycle of a machine learning project in production: problem definition, data collection, feature engineering, model selection, training, evaluation, deployment, monitoring, and retraining. Be thorough.",
            "expected_min_tokens": 400,
        },
    ]
    save_jsonl(prompts, DATASETS_DIR / "long_context_prompts.jsonl")


def setup_manual():
    print("Writing manual eval prompts...")
    prompts = [
        # Reasoning
        {"id": "m_01", "category": "reasoning", "prompt": "If a train travels 120km in 1.5 hours, then stops for 20 minutes, then travels another 80km at the same speed, what is the total journey time?"},
        {"id": "m_02", "category": "reasoning", "prompt": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Show your reasoning step by step."},
        {"id": "m_03", "category": "reasoning", "prompt": "There are 3 boxes: one with apples, one with oranges, one with both. All labels are wrong. You can pick one fruit from one box. How do you label all boxes correctly?"},
        # Coding
        {"id": "m_04", "category": "coding", "prompt": "Write a Python function that finds the two numbers in a list that sum to a target value. Return their indices. Handle edge cases."},
        {"id": "m_05", "category": "coding", "prompt": "Write a SQL query to find the top 5 customers by total purchase value in the last 30 days, from tables: orders(id, customer_id, created_at) and order_items(order_id, price, quantity)."},
        {"id": "m_06", "category": "coding", "prompt": "Explain what this code does and identify any bugs: `def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)`. How would you optimize it?"},
        # Knowledge
        {"id": "m_07", "category": "knowledge", "prompt": "What is the difference between L1 and L2 regularization? When would you use each?"},
        {"id": "m_08", "category": "knowledge", "prompt": "Explain the CAP theorem and give a real-world example of a system that prioritizes each combination."},
        {"id": "m_09", "category": "knowledge", "prompt": "What is the difference between process and thread? Explain with an example where you'd choose one over the other."},
        # Instruction following
        {"id": "m_10", "category": "instruction", "prompt": "List exactly 5 advantages of microservices over monolithic architecture. Use bullet points. Keep each point under 20 words."},
        {"id": "m_11", "category": "instruction", "prompt": "Translate the following to French, then back to English, then note any meaning that was lost: 'The early bird catches the worm but the second mouse gets the cheese.'"},
        {"id": "m_12", "category": "instruction", "prompt": "Respond only in JSON. Keys: 'capital', 'population_millions', 'continent' for the country Germany."},
        # Math
        {"id": "m_13", "category": "math", "prompt": "Solve: 2x² + 5x - 3 = 0. Show all steps."},
        {"id": "m_14", "category": "math", "prompt": "A company grows at 12% per year. How many years to double in size? Use the rule of 72 and verify with exact calculation."},
        {"id": "m_15", "category": "math", "prompt": "What is the probability of rolling at least one 6 in four rolls of a fair die?"},
        # Edge cases
        {"id": "m_16", "category": "edge", "prompt": "What is 0 divided by 0? Explain why mathematicians say this is indeterminate, not undefined."},
        {"id": "m_17", "category": "edge", "prompt": "Write a haiku about quantization in neural networks."},
        {"id": "m_18", "category": "edge", "prompt": "If you had to explain gradient descent to a 10-year-old, what analogy would you use?"},
        {"id": "m_19", "category": "edge", "prompt": "What comes next in this sequence: 1, 1, 2, 3, 5, 8, 13, ___? Now give a different sequence that also starts with 1, 1, 2, 3."},
        {"id": "m_20", "category": "edge", "prompt": "Is this statement true or false, and why: 'All models are wrong but some are useful.'"},
    ]
    save_jsonl(prompts, DATASETS_DIR / "manual_prompts.jsonl")


if __name__ == "__main__":
    print("Setting up benchmark datasets...\n")
    setup_gsm8k()
    setup_humaneval()
    setup_mmlu()
    setup_long_context()
    setup_manual()
    print("\nAll datasets ready.")
