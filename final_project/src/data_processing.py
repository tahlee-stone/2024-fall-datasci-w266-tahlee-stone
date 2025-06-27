# groundedness_project/scripts/data_processing.py

import pandas as pd
import json


def expand_and_sample_jsonl(jsonl_path, output_path, sample_size=500, seed=42):
    """
    Load a QA-style JSONL file, sample N examples, and expand each row
    into two rows: one with a grounded answer and one with a hallucinated one.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = [json.loads(line.strip()) for line in lines]

    df = pd.DataFrame(data)
    sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    expanded_rows = []
    for _, row in sampled.iterrows():
        expanded_rows.append({
            "query": row["question"],
            "context": row["knowledge"],
            "response": row["right_answer"],
            "label": 1
        })
        expanded_rows.append({
            "query": row["question"],
            "context": row["knowledge"],
            "response": row["hallucinated_answer"],
            "label": 0
        })

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.to_csv(output_path, index=False)
    print(f"✅ HaluEval groundedness dataset ({sample_size} examples) saved to '{output_path}'")


if __name__ == "__main__":
    filename = f"../data/halueval_groundedness.csv"
    expand_and_sample_jsonl("../data/qa_data.json", filename)
