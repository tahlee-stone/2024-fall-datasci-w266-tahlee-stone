import os
import sys
import boto3
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report

# Add src to path if running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === Load environment variables ===
load_dotenv()

# === AWS + OpenAI clients ===
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# === Load Prompt Template ===
env = Environment(
    loader=FileSystemLoader("src"), 
    autoescape=select_autoescape()
)
template = env.get_template("groundedness_guardrail.j2")

# === Import modules ===
from comparison_models import run_llm_guardrail_batch, run_bedrock_guardrail_batch
from data_processing import expand_and_sample_jsonl

# === Configuration ===
GUARDRAIL_ID = "egi3t9xv4xej"
GUARDRAIL_VERSION = "1"
MODEL_NAME = "gpt-4"
sample_size = 500
input_json = "inputs/qa_data.json"
output_csv = "inputs/halueval_groundedness.csv"

# === Step 1: Generate Dataset ===
expand_and_sample_jsonl(input_json, output_csv, sample_size=sample_size)

# === Step 2: Load Dataset ===
df = pd.read_csv(output_csv)

# === Step 3: Run LLM Guardrail ===
df = run_llm_guardrail_batch(df.copy(), model=MODEL_NAME, openai_client=openai_client, template=template)

# === Step 4: Run Bedrock Guardrail ===
df = run_bedrock_guardrail_batch(df, guardrail_id=GUARDRAIL_ID, guardrail_version=GUARDRAIL_VERSION, bedrock_client=bedrock_client)

# === Step 5: Prediction and Evaluation  ===
#df["llm_guardrail_pred"] = df["llm_score"].map(lambda x: 1 if str(x).upper() == "PASS" else 0)
#df["bedrock_pred"] = df["grounding_score"].apply(lambda x: 1 if x is not None and x >= 0.5 else 0)
#df["label"] = df["label"].astype(int)

#print("\n📊 Model Evaluation Summary:")
#for model_name in ["llm_guardrail_pred", "bedrock_pred"]:
#    print(f"\n🔍 {model_name}:\n")
#    print(classification_report(df["label"], df[model_name], target_names=["FAIL", "PASS"]))

# === Step 6: Save Results ===
results_path = "results/llm_bedrock_evaluation.csv"
df.to_csv(results_path, index=False)
print(f"\n✅ Evaluation complete. Results saved to '{results_path}'")

