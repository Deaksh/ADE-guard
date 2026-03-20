import argparse
import json

from backend.ade_pipeline import extract_ade_with_severity
from backend.ner_utils import extract_entities
from backend.severity_utils import classify_severity_local


def run(text: str):
    entities = extract_entities(text)
    ade_spans = extract_ade_with_severity(text)
    severity = classify_severity_local(text)

    return {
        "text": text,
        "entities": entities,
        "ade_spans": ade_spans,
        "severity": {
            "label": severity[0] if severity else "Unknown",
            "confidence": severity[1] if severity else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run ADEGuard inference pipeline")
    parser.add_argument("--text", type=str, required=True, help="Input narrative")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON file")
    args = parser.parse_args()

    output = run(args.text)
    payload = json.dumps(output, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
