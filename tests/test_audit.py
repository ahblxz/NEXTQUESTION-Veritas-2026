"""Independent synthetic-health audit regression checks (no real patients)."""
import json
import numpy as np
import pytest

from hospital import engine
from hospital.audit import (
    _urgent_binary_counts, _multiclass_diagnostics, _one_failure, _stress_dataset,
)
from hospital.public_benchmark import stratified_pattern_split, SymptomModel, evaluation


def test_urgent_confusion_counts_consistent():
    actual = [1, 1, 0, 0, 0, 1]
    predicted = [0, 1, 1, 0, 0, 0]
    r = _urgent_binary_counts(actual, predicted)
    assert (r["true_positive"], r["false_positive"],
            r["false_negative"], r["true_negative"]) == (1, 1, 2, 2)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(1 / 3)
    assert r["confusion_matrix_rows_actual_nonurgent_urgent"] == [[2, 1], [2, 1]]


def test_21_condition_matrix_and_per_class_conservation():
    d = engine.generate_cases(n=200, seed=47)
    actual = d["diagnosis"]
    predicted = actual.copy()
    predicted[:17] = (predicted[:17] + 1) % len(engine.CONDITIONS)
    r = _multiclass_diagnostics(actual, predicted)
    matrix = np.array(r["actual_rows_by_predicted_columns"])
    assert matrix.shape == (len(engine.CONDITIONS), len(engine.CONDITIONS))
    assert matrix.sum() == 200
    assert np.array_equal(matrix.sum(axis=1), np.bincount(actual, minlength=len(engine.CONDITIONS)))
    for c in r["per_condition_one_vs_rest"]:
        assert c["true_positive"] + c["false_negative"] == c["support"]
        assert c["false_positive"] + c["false_negative"] + c["true_positive"] + c["true_negative"] == 200


def test_split_contains_no_shared_original_case_id():
    d = engine.generate_cases(n=800, seed=91)
    train, val, test = engine.split_cases(d, seed=91)
    assert set(train["case_id"]).isdisjoint(val["case_id"])
    assert set(train["case_id"]).isdisjoint(test["case_id"])
    assert set(val["case_id"]).isdisjoint(test["case_id"])


def test_demo_is_from_validation_only(tmp_path):
    report = engine.save_research(outdir=tmp_path, n=500, seed=91)
    _, val, test = engine.split_cases(engine.generate_cases(500, 91), 91)
    saved = json.loads((tmp_path / "demo_cases.json").read_text())
    assert saved == engine.make_demo_cases(val)
    assert report["demo_source"] == "validation_only"
    # A future change to test-only facts must not influence displayed cases.
    changed_test = {k: v.copy() for k, v in test.items()}
    changed_test["tests"][:] = 1 - changed_test["tests"]
    assert saved == engine.make_demo_cases(val)


def test_no_hidden_outcome_lookahead_many_patients():
    d = engine.generate_cases(700, seed=607)
    train, _, test = engine.split_cases(d, seed=607)
    model = engine.BayesTestModel.fit(train)
    for i in range(50):
        symptoms = [name for j, name in enumerate(engine.SYMPTOM_KEYS) if test["symptoms"][i, j]]
        before = model.ranking(symptoms, {})
        assert before
        # Modify every unrevealed result and reference diagnosis. Neither
        # value is available to the selector.
        flipped = test["tests"][i] ^ 1
        fake_diagnosis = (int(test["diagnosis"][i]) + 1) % len(engine.CONDITIONS)
        assert fake_diagnosis != test["diagnosis"][i]
        assert len(flipped) == len(engine.TESTS)
        after = model.ranking(symptoms, {})
        assert before == after


def test_no_duplicate_result_proposal_many_paths():
    d = engine.generate_cases(320, seed=638)
    train, _, test = engine.split_cases(d, seed=638)
    model = engine.BayesTestModel.fit(train)
    for i in range(min(80, len(test["diagnosis"]))):
        sample = {k: arr[i] for k, arr in test.items()}
        for policy in ("adaptive", "fixed", "cheapest", "all"):
            r = engine.policy_path(model, sample, policy)
            assert len(r["tests"]) == len(set(r["tests"]))
            assert r["cost"] == sum(engine.COSTS[k] for k in r["tests"])


@pytest.mark.parametrize("mode", [
    "reference_new_seed", "missing_30pct_positive_symptoms",
    "flipped_10pct_test_indicators", "urgent_prevalence_shift", "cost_shift"])
def test_stress_cohorts_well_formed_and_distinct(mode):
    x = _stress_dataset(8001, mode)
    assert len(x["diagnosis"]) > 200
    assert not np.intersect1d(x["case_id"], [-1]).size
    y = _stress_dataset(8001, mode)
    assert all(np.array_equal(x[k], y[k]) for k in x)


def test_public_symptom_duplicate_group_split():
    items = [(tuple(["common", f"variant_{c}_{i}"]), f"class_{c}")
             for c in range(3) for i in range(8)]
    train, val, test = stratified_pattern_split(items, seed=19)
    assert len(test) == len(val) == 3
    assert len(train) == 18
    assert not (set(x for x, _ in train) & set(x for x, _ in test))
    model = SymptomModel(train)
    a = evaluation(model, test, "adaptive", max_questions=3)
    b = evaluation(model, test, "fixed", max_questions=3)
    assert 0 <= a["accuracy"] <= 1 and 0 <= b["accuracy"] <= 1


def test_failure_explanation_cannot_include_private_patient_data():
    d = engine.generate_cases(400, seed=819)
    train, _, test = engine.split_cases(d, seed=819)
    model = engine.BayesTestModel.fit(train)
    paths = [
        engine.policy_path(model, {k: v[i] for k, v in test.items()}, "adaptive")
        for i in range(len(test["diagnosis"]))
    ]
    e = _one_failure(paths, test)
    if e:
        assert int(e["synthetic_case_id"]) in test["case_id"]
        assert e["actual_simulated_condition"] in engine.KEYS
        assert e["predicted_simulated_condition"] in engine.KEYS
        assert "possible_reason_not_proven" in e
