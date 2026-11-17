# roman_urdu_nlp

A reusable, installable Python package for processing raw, noisy Roman Urdu text through a multi-stage NLP pipeline — returning normalized Roman Urdu, Urdu script, and an English translation.

```
Raw Noisy Text → Clean → Detect → Normalize → Correct → Transliterate
                                                        ↘ Translate (English)
```

---

## Features

| Stage | What it does |
|---|---|
| **Cleaner** | Strips emojis, collapses repeated chars, removes punctuation noise |
| **Detector** | Tags code-mixed English vs Roman Urdu segments |
| **Normalizer** | Maps spelling variants to canonical form, expands abbreviations |
| **Corrector** | Edit-distance spell correction + grammar pattern fixes |
| **Transliterator** | Word-map + phoneme-level Roman Urdu → Urdu script |
| **Translator** | Urdu script → English via Google Translate (offline-safe fallback) |

---

## Install

```bash
# Clone the repo
git clone <repo-url>
cd "Roman urdu NLP"

# Install in editable mode (development)
pip install -e .

# Or install with dev/test tools
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+

---

## Quick Start

```python
from roman_urdu_nlp import Pipeline

nlp = Pipeline()
result = nlp("Muje boht bhok lgi h 😂😂")

print(result.normalized)   # "Mujhe bohat bhook lagi hai"
print(result.urdu)         # "مجھے بہت بھوک لگی ہے"
print(result.english)      # "I am very hungry."

# Access intermediate stages
print(result.stages["cleaned"])     # "Muje boht bhok lgi h"
print(result.stages["segments"])    # [('Muje boht bhok lgi h', 'roman_urdu')]
```

---

## More Examples

```python
nlp("mjy neend aa rhi h yaar 😴😴")
# normalized → "mujhe neend aa rahi hai yaar"
# urdu       → "مجھے نیند آ رہی ہے یار"
# english    → "I am feeling sleepy, friend."

nlp("kal exam hai aur maine kuch nhi parha 😰")
# normalized → "kal exam hai aur maine kuch nahi parha"
# urdu       → "کل امتحان ہے اور میں نے کچھ نہیں پڑھا"
# english    → "There is an exam tomorrow and I have not studied."

nlp("meeting kal hai office mein prepare karo")
# normalized → "meeting kal hai office mein prepare karo"
# urdu       → "میٹنگ کل ہے آفس میں prepare کرو"  (code-mixed preserved)
# english    → "The meeting is tomorrow in the office, prepare."
```

---

## Package Structure

```
roman_urdu_nlp/
├── __init__.py          # Public API: Pipeline, PipelineResult
├── cleaner.py           # Stage 1 — emoji/noise removal
├── detector.py          # Stage 2 — language segment tagging
├── normalizer.py        # Stage 3 — spelling normalization + abbreviation expansion
├── corrector.py         # Stage 4 — edit-distance spell correction + grammar fixes
├── transliterator.py    # Stage 5a — Roman Urdu → Urdu script
├── translator.py        # Stage 5b — Urdu script → English
├── pipeline.py          # Integrator — chains all stages
└── data/
    ├── corpus.py        # 50 raw sample sentences
    └── gold_set.py      # 20 gold-standard (raw, normalized, urdu, english) items
tests/
├── test_cleaner.py
├── test_detector.py
├── test_normalizer.py
├── test_corrector.py
├── test_transliterator.py
├── test_translator.py
├── test_pipeline.py
└── test_gold_set.py     # End-to-end evaluation
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Individual module test
pytest tests/test_cleaner.py -v

# End-to-end gold evaluation with output table
pytest tests/test_gold_set.py -v -s

# With coverage
pytest tests/ --cov=roman_urdu_nlp --cov-report=term-missing
```

---

## Architecture Notes

- **No large ML model downloads** — the package uses curated dictionaries + Google Translate API (free tier via `deep-translator`).
- **Offline safe** — the translator module falls back to `"[translation unavailable]"` on network failure.
- **Modular + independently testable** — each module has a single documented function/class, an explicit input/output contract, and its own test file.
- **Mock-first development** — each module was built and tested against a mock of its upstream before real integration.

---

## Dependencies

| Package | Purpose |
|---|---|
| `regex` | Unicode-aware emoji and pattern matching |
| `deep-translator` | Google Translate wrapper (no API key needed) |
| `editdistance` | Levenshtein distance for spell correction |
| `pytest` | Test runner |

---

## Known Limitations

See [EVAL_REPORT.md](EVAL_REPORT.md) for detailed accuracy analysis.

- Normalization dictionary covers ~300 common words; rare words pass through.
- Spell correction limited to edit distance ≤ 1 to avoid over-correction.
- Transliteration of unknown words uses phoneme rules that may produce non-standard spellings.
- Translation quality depends on Google Translate's Urdu model quality.
- Code-mixed English words inside Urdu sentences may not transliterate to correct Urdu script.
