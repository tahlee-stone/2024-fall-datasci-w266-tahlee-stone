import json
import pandas as pd
from src.config import (
    openai_client,
    bedrock_client,
    template
)

# LLM Groundedness Guardrail
def call_llm_guardrail(row, model="gpt-4"):
    prompt = template.render({
        "user_query": row["query"],
        "retrieved_context": row["context"],
        "model_answer": row["answer"]
    })
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_output = response.choices[0].message.content.strip()
        parsed = json.loads(raw_output) if raw_output.startswith("{") else {
            "REASONING": [raw_output],
            "SCORE": "FAIL"
        }
    except Exception as e:
        parsed = {
            "REASONING": [f"Error: {str(e)}"],
            "SCORE": "FAIL"
        }

    return pd.Series({
        "llm_score": parsed.get("SCORE", "FAIL"),
        "llm_reasoning": " | ".join(parsed.get("REASONING", ["Missing reasoning"]))
    })


def run_llm_guardrail_batch(df: pd.DataFrame, model="gpt-4") -> pd.DataFrame:
    results = df.apply(call_llm_guardrail, axis=1)
    return pd.concat([df, results], axis=1)


# AWS Bedrock Guardrail
def call_bedrock_guardrail(row, guardrail_id, guardrail_version):
    payload = {
        "source": "OUTPUT",
        "content": [
            {"text": {"text": row["context"], "qualifiers": ["grounding_source"]}},
            {"text": {"text": row["query"], "qualifiers": ["query"]}},
            {"text": {"text": row["answer"]}}
        ]
    }

    try:
        response = bedrock_client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source=payload["source"],
            content=payload["content"]
        )
        outputs = response.get("outputs", [{}])
        blocked_output = outputs[0].get("text") if outputs else None

        grounding_score = None
        threshold = None
        reason = None
        for a in response.get("assessments", []):
            if "groundingPolicy" in a:
                gp = a["groundingPolicy"]
                grounding_score = gp.get("score")
                threshold = gp.get("threshold")
                reason = gp.get("action")

        return pd.Series({
            "bedrock_action": response.get("action", "UNKNOWN"),
            "grounding_score": grounding_score,
            "grounding_threshold": threshold,
            "grounding_decision_reason": reason,
            "blocked_output_text": blocked_output
        })

    except Exception as e:
        return pd.Series({
            "bedrock_action": "ERROR",
            "grounding_score": None,
            "grounding_threshold": None,
            "grounding_decision_reason": str(e),
            "blocked_output_text": None
        })


def run_bedrock_guardrail_batch(df: pd.DataFrame, guardrail_id: str, guardrail_version: str) -> pd.DataFrame:
    results = df.apply(call_bedrock_guardrail, axis=1, args=(guardrail_id, guardrail_version))
    return pd.concat([df, results], axis=1)