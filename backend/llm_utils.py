# llm_utils.py
import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Explicitly load .env inside backend/
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print("GROQ_API_KEY:", GROQ_API_KEY)
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

# Optional: weak labels reference
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEAK_PATH = os.path.join(BASE_DIR, "data", "weak_labels.json")
weak_labels = {}
if os.path.exists(WEAK_PATH):
    with open(WEAK_PATH, "r") as f:
        data = json.load(f)
        for item in data.get("weak_labels", []):
            weak_labels[item["VAERS_ID"]] = item.get("WeakSeverity", "Unknown")

def classify_severity(text: str, vaers_id: int = None) -> str:
    """
    Classifies adverse event severity using Groq LLM.
    Returns: Mild, Moderate, or Severe
    Fallback: weak_labels.json or Mild
    """
    # fallback if VAERS_ID in weak_labels
    if vaers_id is not None and vaers_id in weak_labels:
        return weak_labels[vaers_id]

    prompt = f"""
    You are a medical assistant. 
    Classify the following adverse event report into one category: Mild, Moderate, or Severe.
    Text: {text}
    Answer with only one word.
    """
    try:
        chat_completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = chat_completion.choices[0].message.content.strip()
        if answer in ["Mild", "Moderate", "Severe"]:
            return answer
        else:
            return "Mild"  # fallback
    except Exception as e:
        print(f"Groq LLM error: {e}")
        return "Mild"
