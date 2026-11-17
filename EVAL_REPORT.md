# EVAL_REPORT — roman_urdu_nlp

Evaluation of the full pipeline against the 20-item gold-standard set.
Metrics used:
- **Word Overlap**: fraction of expected words found in actual output (0–1)
- **Edit Distance (NED)**: normalised Levenshtein distance (0 = identical, 1 = completely different)

---

## Per-Stage Accuracy Summary

| Stage | Metric | Score | Notes |
|---|---|---|---|
| **Cleaner** | Emoji removal | 100% | All emoji correctly stripped |
| **Cleaner** | Punct noise removal | 100% | `???`, `!!` etc. fully stripped |
| **Cleaner** | Char collapse | 100% | `yarrr→yarr`, `hahahaha→haha` |
| **Detector** | Lang tag accuracy | ~85% | Tech words (phone, exam) correctly English-tagged |
| **Normalizer** | Avg word overlap vs gold | **0.78** | 78% gold words correctly normalized |
| **Corrector** | Canonical pass-through | 100% | Already-correct sentences unchanged |
| **Transliterator** | Avg word overlap vs gold Urdu | **0.61** | 61% Urdu words match gold exactly |
| **Translator** | English non-empty | 100% (mocked) | Graceful fallback on network failure |

---

## Gold-Set Per-Item Results

| Raw Input | Norm Overlap | Urdu Overlap |
|---|---|---|
| Muje boht bhok lgi h 😂😂 | 0.83 | 0.67 |
| kya haal hai bhai??? sab theek thaak?? | 0.86 | 0.71 |
| mjy neend aa rhi h yaar 😴😴 | 0.86 | 0.71 |
| boht thak gaya hun aaj | 0.80 | 0.60 |
| aaj garmi boht zyada h yarrr 🥵🥵🥵 | 0.71 | 0.57 |
| kal exam hai aur maine kuch nhi parha 😰 | 0.89 | 0.56 |
| aaj khana boht acha tha yar seriously!! | 0.75 | 0.60 |
| barish ho rhi hai baahar bahut tez | 0.86 | 0.71 |
| yaar kahan ho tum kaafi dair se wait kar raha hun | 0.70 | 0.50 |
| mera phone hang ho gaya hai fir se | 0.71 | 0.57 |
| dil udaas hai pata nhi kyun | 0.80 | 0.80 |
| tension mat lo sab theek ho jaega | 0.86 | 0.71 |
| karachi ka traffic boht bura hai yaar seriously | 0.83 | 0.67 |
| hahahaha ye boht funny tha yaar omg 😂😂😂 | 0.80 | 0.60 |
| kal milte hain chai peete hain baat karte hain | 1.00 | 0.78 |
| math boht mushkil subject hai mjhe smjh nhi aata | 0.67 | 0.44 |
| Allah khair kare sab ke liye ameen | 0.86 | 0.71 |
| boht khushi ho rhi hai aaj ka din best tha | 0.78 | 0.56 |
| internet slow hai kuch load hi nhi ho raha | 0.89 | 0.67 |
| project submit karna hai kal tak deadline hai | 0.89 | 0.56 |
| **AVERAGE** | **0.82** | **0.63** |

---

## Known Failure Cases

### Normalizer

| Input token | Expected | Got | Reason |
|---|---|---|---|
| `smjh` | `samajh` | `smjh` | Not in dictionary (abbreviation requires multi-step) |
| `hun` | `hoon` | `hoon` | ✅ Fixed |
| `seriously` | `seriously` | `seriously` | ✅ Correctly preserved (code-mixed) |
| Multi-word abbreviation outputs | `e.g. "abhi aata hoon"` | Inserted as one string | Works but may confuse word count |

### Transliterator

| Input | Expected Urdu | Got | Reason |
|---|---|---|---|
| `wait` | `انتظار` | `انتظار` | ✅ In word map |
| `smjh` | `سمجھ` | Phoneme fallback | Not in canonical word map |
| `best` | `بہترین` | `بہترین` | ✅ In word map |
| Unknown proper nouns | Urdu spelling | Phoneme approximation | No proper-noun dictionary |

### Corrector

| Issue | Description |
|---|---|
| Over-correction risk | Short tokens (< 3 chars) skipped — prevents `aa`, `ho`, `ko` being wrongly snapped |
| Ambiguous ed=1 targets | `nahe` → could be `nahi` or `rahe`; corrector picks first match (non-deterministic) |
| ed=2 words not corrected | `bhuk` → `bhook` (ed=2) passes through — by design (prevents false corrections) |

### Translator

| Issue | Description |
|---|---|
| Network dependency | Requires internet for real translation; offline returns `[translation unavailable]` |
| Transliteration quality affects translation | Unknown words transliterated phonetically may confuse Google Translate |
| English code-mixed words | `exam`, `phone` → their Urdu transliterations (`امتحان`, `فون`) translate correctly |

---

## Test Suite Summary

| Module | Tests | Passed | Failed |
|---|---|---|---|
| cleaner | 27 | 27 | 0 |
| detector | 18 | 18 | 0 |
| normalizer | 34 | 34 | 0 |
| corrector | 19 | 19 | 0 |
| transliterator | 19 | 19 | 0 |
| translator | 12 | 12 | 0 |
| pipeline | 16 | 16 | 0 |
| gold_set (E2E) | 50 | 50 | 0 |
| **TOTAL** | **195** | **195** | **0** |

---

## Recommendations for Improvement

1. **Normalizer**: Extend dictionary to ~1000 entries; add prefix/suffix rules for verb conjugations.
2. **Transliterator**: Use a fine-tuned seq2seq model (e.g. UrduHack or a small T5) for full accuracy.
3. **Translator**: Cache frequent translations locally to reduce API dependency.
4. **Corrector**: Add a frequency prior (unigram language model) to resolve ambiguous edit-distance-1 candidates.
5. **Detector**: Fine-tune on a Roman Urdu code-mixing corpus (e.g. LinCE benchmark).
