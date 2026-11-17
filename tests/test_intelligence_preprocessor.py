"""
tests/test_intelligence_preprocessor.py — Unit tests for Phase 1 Intelligence Preprocessing Layer (S0-S9).

Covers:
  - Fresh DB table creation (A6)
  - T1: S1 Unicode and whitespace normalization
  - T2: S2 Protected-term masking & sentence-initial capitalization exclusion
  - T3: S3 Whisper artifact drop (no_speech_prob & confidence) and n-gram loop collapse
  - T4: S4 Disfluency removal preserving discourse words & legitimate reduplications
  - T5: S5 Conditional Roman Urdu canonicalization (skipping English prose/homographs)
  - T6: S6 Segment stitching, A4 adapter metrics, and S7 run-on splitting with interpolated timestamps
  - T7: S8/S9 llm_text vs match_text separation
  - T9: Post-S4 Retention safety gate with S1+S3 fallback
  - S0: Numeric & English-word corruption repair via difflib alignment
"""

import os
import re
from types import SimpleNamespace
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("TESTING", "1")


def _make_seg(
    seg_id="s1",
    index=0,
    text="",
    normalized_text=None,
    english_text=None,
    start_time=0.0,
    end_time=4.0,
    no_speech_prob=0.1,
    confidence=0.85,
    language="en",
    translation_source="passthrough",
    chunk_index=None,
):
    return SimpleNamespace(
        id=seg_id,
        segment_index=index,
        chunk_index=chunk_index,
        text=text,
        normalized_text=normalized_text if normalized_text is not None else text,
        english_text=english_text,
        start_time=start_time,
        end_time=end_time,
        no_speech_prob=no_speech_prob,
        confidence=confidence,
        language=language,
        translation_source=translation_source,
    )


def test_a6_fresh_db_creates_intelligence_views_table():
    """Confirm Base.metadata.create_all creates `intelligence_views` when main/preprocessor are imported."""
    import app.main  # noqa: F401
    from app.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "intelligence_views" in tables
    cols = {c["name"] for c in inspector.get_columns("intelligence_views")}
    assert {"id", "recording_id", "transcript_hash", "preprocessing_version", "segments_json", "full_llm_text", "full_match_text"} <= cols


def test_a1_s0_repairs_numeric_and_english_corruptions():
    """T0/A1: S0 restores numbers (20,000 -> 20,00) and English words (cost->dost, basic->basis, two->wo)."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            seg_id="seg_0",
            index=0,
            text="Suppose your cost was two lakh and 20,000 rupees in this basic model?",
            normalized_text="Suppose your dost was wo lakh and 20,00 rupees in this basis model",
            english_text="Paraphrased translator output.",
            translation_source="local_llm",
            language="en",
        )
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec1", segs, write_diff_log=False)
    assert len(view.segments) == 1
    out = view.segments[0]
    assert out.source_choice == "repaired_english_dominant"
    assert "cost" in out.llm_text and "dost" not in out.llm_text
    assert "20,000" in out.llm_text and not re.search(r"\b20,00\b", out.llm_text)
    assert "basic" in out.llm_text and "basis" not in out.llm_text
    assert "two" in out.llm_text
    assert out.llm_text.endswith("?")


def test_a1_s0_non_english_uses_english_text_only_when_numbers_match():
    """Non-English segments use english_text only if all numbers survive; else fallback to repaired source."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    # Bad translation mangles number 50,000 -> falls back to repaired source
    seg_mismatch = _make_seg(
        seg_id="s_bad_num",
        index=0,
        text="ye 50,000 rupay ka kharcha hai bhai",
        normalized_text="ye 50,00 rupay ka kharcha hai bhai",
        english_text="This expenditure is five hundred rupees brother.",
        language="ur",
        translation_source="local_llm",
    )
    # Valid translation preserves number 50,000
    seg_good_num = _make_seg(
        seg_id="s_good_num",
        index=1,
        text="ye 50,000 rupay ka munafa hai",
        normalized_text="ye 50,000 rupay ka munafa hai",
        english_text="This is a profit of 50,000 rupees.",
        start_time=5.0,
        end_time=9.0,
        language="ur",
        translation_source="local_llm",
    )
    view = IntelligencePreprocessor.build_view_in_memory("rec_num", [seg_mismatch, seg_good_num], write_diff_log=False)
    assert view.segments[0].source_choice == "repaired_fallback_number_mismatch"
    assert "50,000" in view.segments[0].llm_text
    assert view.segments[1].source_choice == "english_text_verified"
    assert "50,000" in view.segments[1].llm_text


def test_t1_s1_unicode_and_whitespace_normalization():
    """T1: NFKC normalization, zero-width char removal, smart quote/dash unification, space collapsing."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            text="  \ufeffThis\u200b  is  “Operating\u200d Systems” —  very   important…  ",
            english_text=None,
        )
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_t1", segs, write_diff_log=False)
    llm = view.segments[0].llm_text
    assert "\u200b" not in llm and "\u200d" not in llm and "\ufeff" not in llm
    assert '"Operating Systems"' in llm
    assert " - " in llm
    assert "..." in llm
    assert "  " not in llm


def test_t2_s2_protected_terms_and_no_sentence_start_mask():
    """T2: Runtime topic_hint, acronyms, CamelCase/snake_case, units, mid-sentence caps protected; sentence-start ignored."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    stats = {"rule_counts": {"S2_protected_terms_masked": 0}}
    txt = "Today we study CPU scheduling in Operating Systems using page_size 50ms and FinTech API."
    masked, tmap = IntelligencePreprocessor.s2_mask_protected_terms(txt, txt, "Operating Systems", stats)

    protected_vals = set(tmap.values())
    assert "CPU" in protected_vals
    assert "API" in protected_vals
    assert "Operating Systems" in protected_vals
    assert "page_size" in protected_vals
    assert "50ms" in protected_vals
    assert "FinTech" in protected_vals
    # Sentence-initial 'Today' must NOT be falsely protected
    assert "Today" not in protected_vals
    assert masked.startswith("Today")


def test_t3_s3_whisper_artifact_and_loop_collapse():
    """T3: Drops silence+low_conf (using DB fields no_speech_prob & confidence), drops hallucinations, collapses 3+ n-grams."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        # Dropped: nsp > 0.6 and confidence < 0.3679
        _make_seg(seg_id="drop_ac", index=0, text="random noise words", no_speech_prob=0.75, confidence=0.20),
        # Kept: nsp > 0.6 BUT confidence >= 0.3679 (protects real short segments like 'Gantt chart')
        _make_seg(seg_id="keep_gantt", index=1, text="Gantt chart shows CPU timeline.", start_time=5.0, end_time=9.0, no_speech_prob=0.62, confidence=0.48),
        # Dropped: hallucination boilerplate
        _make_seg(seg_id="drop_hal", index=2, text="Thanks for watching!", start_time=10.0, end_time=12.0, no_speech_prob=0.1, confidence=0.9),
        # Loop collapsed: 3+ n-gram repetition -> kept once
        _make_seg(
            seg_id="keep_loop",
            index=3,
            text="The cash flow statement shows flows and outflows flows and outflows flows and outflows clearly.",
            start_time=15.0,
            end_time=22.0,
        ),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_t3", segs, write_diff_log=False)
    assert view.rule_stats["rule_counts"]["S3_dropped_acoustic"] == 1
    assert view.rule_stats["rule_counts"]["S3_dropped_hallucination"] == 1
    assert view.rule_stats["rule_counts"]["S3_loop_collapsed"] == 1
    all_src_ids = [sid for s in view.segments for sid in s.source_segment_ids]
    assert "drop_ac" not in all_src_ids
    assert "drop_hal" not in all_src_ids
    assert "keep_gantt" in all_src_ids
    loop_seg = next(s for s in view.segments if "keep_loop" in s.source_segment_ids)
    assert loop_seg.llm_text.lower().count("flows and outflows") == 1


def test_t4_s4_disfluency_cleanup_preserves_reduplication():
    """T4: Pure fillers stripped, meaningful discourse kept, function stutters collapsed, reduplications kept."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            text="Um basically ye ye theek hai matlab bohat bohat zaroori concept hai uhhh.",
            language="ur",
            translation_source=None,
        )
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_t4", segs, write_diff_log=False)
    llm = view.segments[0].llm_text
    assert "um" not in llm.lower().split()
    assert "uhhh" not in llm.lower()
    assert "basically" in llm.lower()
    assert "matlab" in llm.lower()
    assert "theek" in llm.lower()
    assert "ye ye" not in llm.lower()
    assert "bohat bohat" in llm.lower()


def test_t5_s5_conditional_roman_urdu_canonicalization():
    """T5: S5 canonicalizes Roman Urdu variants (kya/kia, nhi/nahi) while leaving English homographs (per, cost) intact."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    ur_seg = _make_seg(
        seg_id="ur1",
        index=0,
        text="ye kch nhi h aur kia haal hai",
        normalized_text="ye kch nhi h aur kia haal hai",
        language="ur",
        translation_source=None,
    )
    en_seg = _make_seg(
        seg_id="en1",
        index=1,
        text="We process 100 transactions per second at low cost.",
        start_time=10.0,
        end_time=15.0,
        language="en",
        translation_source="passthrough",
    )
    view = IntelligencePreprocessor.build_view_in_memory("rec_t5", [ur_seg, en_seg], write_diff_log=False)
    ur_llm = view.segments[0].llm_text.lower()
    en_llm = view.segments[1].llm_text.lower()
    assert "kuch" in ur_llm and "nahi" in ur_llm and "hai" in ur_llm
    assert "per second" in en_llm and "par second" not in en_llm


def test_t6_s6_stitching_adapter_metrics_and_s7_runon_splitting():
    """T6 & A4: Stitches across broken ASR seams (<3s), respects >3s gap, carries min conf/nsp, splits >60w run-ons."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            seg_id="a", index=0, start_time=0.0, end_time=3.0,
            confidence=0.80, no_speech_prob=0.10,
            text="Accounting profit measures revenue earned during the period",
        ),
        _make_seg(
            seg_id="b", index=1, start_time=3.4, end_time=7.0,
            confidence=0.55, no_speech_prob=0.05,
            text="whereas cash flow tracks actual money entering the bank.",
        ),
        # Gap > 3.0s -> must NOT merge (70+ unique words to test S7 run-on splitting cleanly)
        _make_seg(
            seg_id="c", index=2, start_time=12.0, end_time=24.0,
            confidence=0.90, no_speech_prob=0.02,
            text=(
                "Companies operating in competitive international markets regularly evaluate quarterly cash reserves "
                "alongside projected capital expenditures so financial managers and treasury analysts can anticipate working capital gaps "
                "before raw material suppliers demand immediate settlement of overdue invoices and monthly payroll obligations remain "
                "fully funded across seasonal downturns and unexpected customer payment delays because sudden liquidity "
                "shortages can severely disrupt daily industrial operations even when annual accounting income statements display strong net profits."
            ),
        ),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_t6", segs, write_diff_log=False)
    adapters = view.to_adapters()

    # Check merged segment ("a" + "b")
    merged = adapters[0]
    assert merged.source_segment_ids == ["a", "b"]
    assert merged.start_time == 0.0 and merged.end_time == 7.0
    assert merged.confidence == 0.55        # min(0.80, 0.55) per A4
    assert merged.no_speech_prob == 0.05    # min(0.10, 0.05) per A4
    assert "Accounting profit" in merged.llm_text and "entering the bank." in merged.llm_text

    # Check >60-word run-on ("c") split into >= 2 units preserving source_segment_ids & timestamps
    split_units = [u for u in adapters if "c" in u.source_segment_ids]
    assert len(split_units) >= 2
    assert split_units[0].start_time == 12.0
    assert split_units[-1].end_time == 24.0
    assert split_units[0].end_time < split_units[-1].end_time


def test_t7_s8_s9_llm_vs_match_text_separation():
    """T7: llm_text preserves stopwords, original casing & punctuation; match_text is lowercase without stopwords."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            text="What is the primary difference between Accrual Accounting and Cash Flow?",
            normalized_text="What is the primary difference between Accrual Accounting and Cash Flow?",
        )
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_t7", segs, topic_hint="Accrual Accounting", write_diff_log=False)
    seg = view.segments[0]
    assert "What is the primary" in seg.llm_text
    assert seg.llm_text.endswith("?")
    assert "Accrual Accounting" in seg.llm_text

    # match_text removes stopwords and lowercases
    assert "the" not in seg.match_text.split()
    assert "what" not in seg.match_text.split()
    assert "between" not in seg.match_text.split()
    assert "accrual" in seg.match_text and "accounting" in seg.match_text and "cash" in seg.match_text


def test_t9_retention_gate_post_s4_and_s1_s3_fallback(monkeypatch):
    """T9 & A3: Retention gate compares llm_text to post-S4; falls back to S1+S3 output if retention < 0.80."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    # First verify S3 loop removal does NOT trigger fallback when S4 keeps real content
    segs_loop = [
        _make_seg(
            seg_id="loop1",
            text="Thank you. Process scheduling scheduling scheduling scheduling scheduling allocates CPU resources cleanly.",
        )
    ]
    view_ok = IntelligencePreprocessor.build_view_in_memory("rec_ok", segs_loop, write_diff_log=False)
    assert view_ok.segments[0].is_fallback is False
    assert view_ok.segments[0].retention_ratio >= 0.80

    # Simulate an over-pruning step between S4 and S8 (dropping content tokens after S4)
    def fake_s5(text, is_en_dom, trans_src, stats, **kwargs):
        return "CPU resources."  # Artificially drops most content tokens!

    monkeypatch.setattr(IntelligencePreprocessor, "s5_roman_urdu_canonicalize", classmethod(lambda cls, *a, **kw: fake_s5(*a, **kw)))

    segs_drop = [
        _make_seg(
            seg_id="fb1",
            text="Thanks for watching! Uh CPU scheduling allocates processor cycles efficiently to ready processes.",
        )
    ]
    # Note: Segment starts with real content, has "uh" removed by S4, and mocked S5 drops content.
    # Fallback must restore S1+S3 output (including content words, without trailing stutter if removed or cleanly from S3).
    segs_drop2 = [
        _make_seg(
            seg_id="fb2",
            text="CPU scheduling allocates processor cycles efficiently to ready processes cycles efficiently cycles efficiently cycles efficiently.",
        )
    ]
    view_fb = IntelligencePreprocessor.build_view_in_memory("rec_fb", segs_drop2, write_diff_log=False)
    assert view_fb.segments[0].is_fallback is True
    assert view_fb.segments[0].retention_ratio < 0.80
    # Restored from S1+S3: content preserved
    assert "allocates processor cycles efficiently to ready processes" in view_fb.segments[0].llm_text


def test_f1_unsafe_english_text_rejected_and_trust_flagged():
    """F1: Rejects english_text on low chunk conf (<0.3679), <6 tokens, >40% diff, or 0 overlap; sets trust='low'."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        # Chunk 14 low-confidence garbage translated to "Let us be friends." / "By this tree alone."
        _make_seg(
            seg_id="c14_1", index=0, chunk_index=14, confidence=0.1012, no_speech_prob=0.25,
            start_time=0.0, end_time=2.0,
            text="bßniisni,", normalized_text="bßniisni,", english_text="Let us be friends.",
            language="ur", translation_source="local_llm",
        ),
        _make_seg(
            seg_id="c14_2", index=1, chunk_index=14, confidence=0.1012, no_speech_prob=0.25,
            start_time=6.0, end_time=8.0,
            text="bhi sa Treeal,", normalized_text="bhi sa treeal,", english_text="By this tree alone.",
            language="ur", translation_source="local_llm",
        ),
        # Short greeting (<6 tokens) + zero content overlap even with good confidence
        _make_seg(
            seg_id="fintech_0", index=2, chunk_index=0, confidence=0.5897, no_speech_prob=0.16,
            start_time=12.0, end_time=15.0,
            text="Assalamu alaikum everyone", normalized_text="Assalamu alaikum everyone",
            english_text="Good morning, all!", language="ur", translation_source="local_llm",
        ),
        # >6 tokens, good confidence, >=1 content overlap ("accounting"), <=40% length diff -> accepted & high trust
        _make_seg(
            seg_id="good_tr", index=3, chunk_index=1, confidence=0.78, no_speech_prob=0.05,
            start_time=20.0, end_time=25.0,
            text="ye accounting ka bohat zaroori usool hota hai",
            normalized_text="ye accounting ka bohat zaroori usool hota hai",
            english_text="This is a very important principle of accounting.",
            language="ur", translation_source="local_llm",
        ),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_f1", segs, write_diff_log=False)
    by_id = {s.source_segment_ids[0]: s for s in view.segments}

    assert "Let us be friends" not in by_id["c14_1"].llm_text
    assert by_id["c14_1"].trust == "low"
    assert by_id["c14_1"].source_choice.startswith("repaired_fallback_")

    assert "By this tree alone" not in by_id["c14_2"].llm_text
    assert by_id["c14_2"].trust == "low"

    assert "Good morning" not in by_id["fintech_0"].llm_text
    assert by_id["fintech_0"].trust == "low"
    assert by_id["fintech_0"].source_choice == "repaired_fallback_short_segment"

    assert by_id["good_tr"].source_choice == "english_text_verified"
    assert by_id["good_tr"].trust == "high"


def test_f2_conservative_lexicon_revert_and_unsure_logging():
    """F2: Corrupted tokens in roman_urdu_nlp lexicon are kept + logged UNSURE unless 2-token English neighbors confirm."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        # Confirmed English neighbors around 'dost' (<-- 'cost'): should auto-revert to 'cost'
        _make_seg(
            seg_id="confirmed", index=0, start_time=0.0, end_time=4.0,
            text="Suppose your cost is very high today",
            normalized_text="Suppose your dost is very high today",
            language="en",
        ),
        # Unconfirmed neighbors ('ho tt ... aaj') around Roman Urdu lexicon token 'uff'/'ke': must NOT auto-revert -> UNSURE
        _make_seg(
            seg_id="unconfirmed", index=1, start_time=10.0, end_time=14.0,
            text="Reduce ho off please aaj if possible",
            normalized_text="Reduce ho uff please aaj ke possible",
            language="en",
        ),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_f2", segs, write_diff_log=False)
    assert "cost" in view.segments[0].llm_text and "dost" not in view.segments[0].llm_text
    unsure = view.rule_stats["s0_unsure_reverts"]
    assert len(unsure) >= 1
    assert any(u["corrupt"] in ("uff", "ke") and u["context_before"] is not None for u in unsure)
    assert "uff" in view.segments[1].llm_text.lower() or "ke" in view.segments[1].llm_text.lower()


def test_f3_negations_and_quantifiers_kept_in_match_text():
    """F3: Negations & quantifiers (not, no, never, only, all, most, don't, nahi) are preserved in match_text."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            seg_id="neg1", index=0, start_time=0.0, end_time=5.0,
            text="This statement is not true and not necessarily exhaustive for all models.",
        ),
        _make_seg(
            seg_id="neg2", index=1, start_time=10.0, end_time=15.0,
            text="Most companies don't recognize revenue without cash, aur ye sahi nahi hai.",
            language="ur",
        ),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_f3", segs, write_diff_log=False)
    m0 = view.segments[0].match_text.split()
    assert "not" in m0 and "true" in m0 and "all" in m0
    # Ensure regular stopwords are still stripped
    assert "is" not in m0 and "for" not in m0 and "this" not in m0

    m1 = view.segments[1].match_text.split()
    assert "most" in m1 and "without" in m1 and "don't" in m1 and "nahi" in m1


def test_f5_timestamps_monotonic_and_non_overlapping():
    """F5: Output timestamps satisfy start_time[i+1] >= end_time[i] even with overlapping raw Whisper chunks."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(seg_id="c0", index=0, start_time=94.66, end_time=98.94, text="First chunk ends with a period."),
        _make_seg(seg_id="c1", index=1, start_time=97.94, end_time=103.94, text="Second chunk overlaps in raw timestamps."),
        _make_seg(seg_id="c2", index=2, start_time=102.50, end_time=106.00, text="Third chunk also overlaps."),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_f5", segs, write_diff_log=False)
    for a, b in zip(view.segments, view.segments[1:]):
        assert b.start_time >= a.end_time, f"Overlap detected: {a.end_time} > {b.start_time}"
    for s in view.segments:
        assert s.end_time >= s.start_time


def test_phase2_s6_does_not_merge_across_trust_boundary():
    """Phase 2: Low-trust chunks must stay isolated from adjacent high-trust segments during S6 stitching."""
    from app.services.intelligence_preprocessor import IntelligencePreprocessor

    segs = [
        _make_seg(
            seg_id="low_c14", index=0, chunk_index=14, confidence=0.10, no_speech_prob=0.25,
            start_time=0.0, end_time=3.0,
            text="bßniisnii,", normalized_text="bßniisnii,", english_text="Let us be friends.",
            language="ur", translation_source="local_llm",
        ),
        _make_seg(
            seg_id="high_c15", index=1, chunk_index=15, confidence=0.85, no_speech_prob=0.05,
            start_time=3.2, end_time=8.0,
            text="cash inflows is negative 50,000 and company profit is weak.",
            normalized_text="cash inflows is negative 50,000 and company profit is weak.",
            language="en",
        ),
    ]
    view = IntelligencePreprocessor.build_view_in_memory("rec_trust_boundary", segs, write_diff_log=False)
    assert len(view.segments) == 2
    assert view.segments[0].trust == "low"
    assert view.segments[1].trust == "high"
    assert "50,000" in view.segments[1].llm_text
    assert "bßniisnii" not in view.segments[1].llm_text

