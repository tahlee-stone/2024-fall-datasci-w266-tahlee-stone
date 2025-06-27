# combined_model_evaluation.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import boto3
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["query", "context", "response", "label"])
    return df

def run_cross_encoder(df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize(row):
        inputs = f"[QUESTION] {row['query']} [CONTEXT] {row['context']} [RESPONSE] {row['response']}"
        return tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

    predictions = []
    for _, row in df.iterrows():
        inputs = tokenize(row)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

    return predictions

def run_biencoder(df):
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    preds = []
    for _, row in df.iterrows():
        ctx = encoder.encode(row["context"], convert_to_tensor=True)
        rsp = encoder.encode(row["response"], convert_to_tensor=True)
        sim = util.cos_sim(ctx, rsp).item()
        preds.append(1 if sim > 0.7 else 0)
    return preds

def run_llm_verifier(df, model="gpt-4", api_key=None):
    openai.api_key = api_key
    preds = []
    for _, row in df.iterrows():
        prompt = f"You are a factuality evaluator. Given the user query, retrieved context, and LLM response, decide whether the response is grounded in the context. Reply only with PASS or FAIL.\nQuery: {row['query']}\nContext: {row['context']}\nResponse: {row['response']}\nAnswer:"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = response.choices[0].message.content.strip().upper()
            preds.append(1 if answer == "PASS" else 0)
        except Exception:
            preds.append(0)
    return preds

def run_bedrock_guardrail(df, region="us-east-1"):
    client = boto3.client("bedrock-runtime", region_name=region)
    preds = []
    for _, row in df.iterrows():
        payload = {
            "user_query": row["query"],
            "retrieved_context": row["context"],
            "model_answer": row["response"]
        }
        try:
            response = client.invoke_model(
                modelId="amazon.guardrails.groundedness",
                body=json.dumps(payload),
                contentType="application/json"
            )
            parsed = json.loads(response["body"].read())
            preds.append(1 if parsed.get("SCORE", 0) else 0)
        except Exception:
            preds.append(0)
    return preds

def evaluate_all(y_true, model_outputs):
    results = {}
    Path("results").mkdir(exist_ok=True)
    for model_name, preds in model_outputs.items():
        print(f"\n📊 Evaluation for {model_name}:")
        report = classification_report(y_true, preds, target_names=["FAIL", "PASS"], output_dict=True)
        print(classification_report(y_true, preds, target_names=["FAIL", "PASS"]))
        results[model_name] = report

        # Save classification report to CSV
        report_df = pd.DataFrame(report).transpose()
        report_path = f"results/{model_name}_classification_report.csv"
        report_df.to_csv(report_path)
        print(f"📄 Saved report to {report_path}")

        # Save confusion matrix
        cm = confusion_matrix(y_true, preds)
        cm_df = pd.DataFrame(cm, index=["FAIL", "PASS"], columns=["Predicted FAIL", "Predicted PASS"])
        cm_path = f"results/{model_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        print(f"📄 Saved confusion matrix to {cm_path}")

    return results
def plot_comparison(results):
    f1_scores = {k: v["weighted avg"]["f1-score"] for k, v in results.items()}
    df = pd.DataFrame.from_dict(f1_scores, orient="index", columns=["F1 Score"])
    df = df.sort_values("F1 Score", ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=df.index, y="F1 Score", data=df)
    plt.title("F1 Score Comparison Across Models")
    plt.xticks(rotation=30)
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/model_comparison.png")
    print("📈 Saved model comparison chart to results/model_comparison.png")

if __name__ == "__main__":
    df = load_dataset("data/groundedness_dataset.csv")
    y_true = df["label"].tolist()

    outputs = {
        "CrossEncoder": run_cross_encoder(df, "models/cross_encoder"),
        "BiEncoder": run_biencoder(df),
        "LLMVerifier": run_llm_verifier(df, api_key="your-openai-key"),
        "BedrockGuardrail": run_bedrock_guardrail(df)
    }

    evaluation = evaluate_all(y_true, outputs)
    plot_comparison(evaluation)
