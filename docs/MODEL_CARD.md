# Model Card: ADEGuard

## Model Overview
- Task: ADE + DRUG span extraction and severity classification
- NER backbone: BioBERT (`dmis-lab/biobert-base-cased-v1.1`)
- Severity backbone: RoBERTa-base (replace with BioBERT if required)

## Intended Use
- Assist safety teams with early signal detection from VAERS narratives.
- Support downstream analytics and clustering.

## Limitations
- Performance depends on quality of gold data.
- Severity model trained on weak labels; may be noisy.
- VAERS narratives are self-reported and may contain inaccuracies.

## Data
- VAERS 2020–2025 reports (currently 2025 in this repo).
- Optional COVID-only filtering for evaluation consistency.

## Performance
- TODO: Add precision/recall/F1 by entity type on a held-out gold COVID test set.
- TODO: Add severity classification metrics.

## Ethical Considerations
- Not a diagnostic tool.
- Use for triage and monitoring only.

## Training Details
- See `/Users/deakshshetty/Documents/ADE-Guard/backend/train_ner.py`
- See `/Users/deakshshetty/Documents/ADE-Guard/backend/train_severity.py`
