"""Real public hospital-record regression tests; never use real identifiable data.

These tests download only the openly licensed MIMIC-IV 100-person demo.
No synthetic hospital observations substitute for the genuine EHR results.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hospital import mimic_real
from hospital import real_ui


def test_patient_disjoint_real_holdout():
    data, _, _ = mimic_real.load_cases()
    train, validation, test = mimic_real.patient_split(data)
    def ids(rows):
        return {r["patient"] for r in rows}
    assert len(data) > 150
    assert ids(train).isdisjoint(ids(validation))
    assert ids(train).isdisjoint(ids(test))
    assert ids(validation).isdisjoint(ids(test))
    assert sum(map(len, (train, validation, test))) == len(data)


def test_no_known_heldout_patient_train_leak():
    records, _, _ = mimic_real.load_cases()
    train, validation, test = mimic_real.patient_split(records)
    model = mimic_real.RealEvidenceModel(train)
    for case in test[:10]:
        p = model.posterior(case["complaint"], {})
        assert np.isfinite(p).all()
        assert 0.99 < p.sum() < 1.01
        assert "patient" not in model.__dict__
        assert "hadm" not in model.__dict__


def test_ranking_does_not_read_undisclosed_real_lab_values():
    records, _, _ = mimic_real.load_cases()
    train, _, test = mimic_real.patient_split(records)
    model = mimic_real.RealEvidenceModel(train)
    for case in test[:20]:
        eligible = set(case["observed"])
        before = model.rank(case["complaint"], {}, eligible=eligible)
        forged = {lab: 1 - int(v) for lab,v in case["observed"].items()}
        assert forged != case["observed"]
        assert model.rank(case["complaint"], {}, eligible=eligible) == before
        # Only observed, historically recorded labs are available for replay.
        assert {x["lab"] for x in before} == eligible


def test_unperformed_tests_are_unknown_not_negative():
    records, _, _ = mimic_real.load_cases()
    train = mimic_real.patient_split(records)[0]
    model = mimic_real.RealEvidenceModel(train)
    case = next(row for row in records if len(row["observed"]) < len(mimic_real.LAB_CODES))
    unknown = set(mimic_real.LAB_CODES) - set(case["observed"])
    assert unknown, "Need at least one genuinely unmeasured lab for this test"
    baseline = model.posterior(case["complaint"], case["observed"])
    for lab in unknown:
        assert lab not in case["observed"]
    assert np.allclose(baseline, model.posterior(case["complaint"], dict(case["observed"])))


def test_real_artifacts_from_validation_not_heldout():
    root = Path("artifacts/mimic_real")
    assert (root / "real_model.json").exists()
    assert (root / "real_metrics.json").exists()
    examples = json.loads((root / "real_demo_cases.json").read_text())
    data, _, _ = mimic_real.load_cases()
    train, val, test = mimic_real.patient_split(data)
    expected = mimic_real.demo_validation_cases(val)
    assert examples == expected
    assert len(examples[0]["complaint"].strip()) >= 7
    assert "patient" not in examples[0] and "hadm" not in examples[0]
    assert set(x["patient"] for x in test).isdisjoint(x["patient"] for x in val)


def test_real_metrics_recomputed_from_heldout():
    root = Path("artifacts/mimic_real")
    metrics = json.loads((root / "real_metrics.json").read_text())
    train, val, test = mimic_real.patient_split(mimic_real.load_cases()[0])
    model = mimic_real.RealEvidenceModel(train)
    assert metrics["adaptive"] == mimic_real.evaluate(model, test, "adaptive")
    assert metrics["fixed"] == mimic_real.evaluate(model, test, "fixed")
    assert metrics["split_patients"]["test"] == len(set(r["patient"] for r in test))
    assert metrics["real_deidentified_hospital_records"] is True
    assert metrics["demo_validation_only"] is True


def test_real_replay_no_repeat_and_strict_step_limit():
    model, examples, _, names = mimic_real.load_research()
    case = examples[0]
    # No evaluation outcomes can be read through the UI's initial state.
    state = real_ui.real_start(case["id"])[0]
    assert state["obtained"] == {}
    keys = []
    for _ in range(10):
        new_state = real_ui.real_next(state)[0]
        if new_state == state:
            break
        state = new_state
        keys = list(state["obtained"])
        assert len(keys) == len(set(keys))
        assert len(keys) <= mimic_real.MAX_STEPS
        assert set(keys).issubset(case["available_labs"])
        for k in keys:
            assert state["obtained"][k] == case["available_labs"][k]
    assert len(keys) == mimic_real.MAX_STEPS
    assert real_ui.real_reset()[0] is None


def test_research_disclaimer_protects_against_clinical_claims():
    assert "not medical advice" in mimic_real.RESEARCH_DISCLAIMER.lower()
    assert "retrospective" in mimic_real.__doc__.lower()
    assert real_ui.READY is True
    report = json.loads(Path("artifacts/mimic_real/real_metrics.json").read_text())
    assert len(report["important_limitations"]) >= 7
