# Evaluation Report

## Scope
- COVID-19 VAERS narratives only
- Gold standard ADE/DRUG spans

## NER Metrics (ADE/DRUG)
Run:
```
COVID_ONLY=1 NER_LABELS=ADE,DRUG \
python3 /Users/deakshshetty/Documents/ADE-Guard/backend/scripts/evaluate_ner.py
```

Results:
- Precision: TBD
- Recall: TBD
- F1: TBD

## Severity Metrics
Run after training:
```
COVID_ONLY=1 \
python3 /Users/deakshshetty/Documents/ADE-Guard/backend/train_severity.py
```

Results:
- Macro-F1: TBD
- Per-class F1: TBD

## Notes
- Ensure COVID-only filtering is consistent across train/val/test.
- Use the same model checkpoints across evaluation and inference.
