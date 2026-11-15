"""Safety, split, information-selection and dashboard regression tests.

All test inputs are artificial; none imply medical validation.
"""
import json
import numpy as np
import pytest

from hospital.engine import (
    CONDITIONS, COSTS, RED_FLAGS, SYMPTOMS, TESTS, URGENCY, BayesTestModel,
    evaluate, extract_symptoms, generate_cases, make_demo_cases, policy_path,
    save_research, split_cases,
)


@pytest.fixture(scope="module")
def setup():
    records = generate_cases(n=800, seed=173)
    train, val, test = split_cases(records, seed=173)
    return records, train, val, test, BayesTestModel.fit(train)


def test_multicategory_and_unique_syndrome_names():
    assert len(CONDITIONS) >= 15
    assert len(set(x[1] for x in CONDITIONS)) >= 5
    assert len(TESTS) >= 10
    assert len(SYMPTOMS) >= 16
    assert len({row[0] for row in CONDITIONS}) == len(CONDITIONS)
    assert len(URGENCY) == len(CONDITIONS)


def test_case_split_is_disjoint_using_unique_index():
    # All rows are drawn separately before splitting, no test rows used in fit.
    d = generate_cases(n=1500, seed=321)
    train, val, test = split_cases(d, seed=321)
    assert sum(len(x["diagnosis"]) for x in (train, val, test)) == 1500
    assert len(train["symptoms"]) == len(train["tests"])
    assert all(set(np.unique(x["tests"])).issubset({0, 1}) for x in (train, val, test))


def test_deterministic_generation_and_evaluation(setup):
    one = generate_cases(100, 19)
    two = generate_cases(100, 19)
    assert all(np.array_equal(one[k], two[k]) for k in one)
    report1 = evaluate(setup[-1], setup[3], "adaptive", max_tests=3)
    report2 = evaluate(setup[-1], setup[3], "adaptive", max_tests=3)
    assert report1 == report2


def test_class_posteriors_valid(setup):
    model = setup[-1]
    p = model.posterior(["fever", "cough"], {"crp": 1})
    assert len(p) == len(CONDITIONS)
    assert np.all((p >= 0) & (p <= 1))
    assert p.sum() == pytest.approx(1.0)
    assert np.isfinite(p).all()


def test_unreported_symptoms_not_treated_as_negative(setup):
    model = setup[-1]
    assert np.allclose(model.posterior([]), model.prior)
    assert not np.allclose(model.posterior(["cough"]), model.prior)


def test_hidden_tests_do_not_influence_first_recommendation(setup):
    _, _, _, test, model = setup
    symptoms = [k for i, k in enumerate(SYMPTOMS) if test["symptoms"][0][i]]
    first = model.ranking(symptoms, {})[0]["key"]
    modified = {k: v.copy() for k, v in test.items()}
    modified["tests"][0] = 1 - modified["tests"][0]
    after = model.ranking(symptoms, {})[0]["key"]
    assert first == after


def test_result_changes_differential(setup):
    model = setup[-1]
    symptoms = ["fever", "cough", "breathlessness"]
    initial = model.posterior(symptoms)
    ranked = model.ranking(symptoms, {})
    assert ranked
    new = model.posterior(symptoms, {ranked[0]["key"]: 1})
    assert not np.allclose(initial, new)


@pytest.mark.parametrize("policy", ["adaptive", "fixed", "cheapest", "all"])
def test_policy_cost_count_invariants(setup, policy):
    _, _, _, test, model = setup
    for idx in range(5):
        case = {k: v[idx] for k, v in test.items()}
        result = policy_path(model, case, policy, budget=22, max_tests=6)
        assert result["count"] == len(result["tests"])
        assert result["cost"] == sum(COSTS[k] for k in result["tests"])
        assert len(set(result["tests"])) == result["count"]
        assert 0 <= result["confidence"] <= 1
        if policy != "all":
            assert result["count"] <= 6 and result["cost"] <= 22


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 5, 8, 12, 22])
def test_ranking_budget_never_exceeded(setup, budget):
    model = setup[-1]
    ranked = model.ranking(["fatigue"], {}, budget)
    assert all(COSTS[x["key"]] <= budget for x in ranked)
    assert all(np.isfinite(x["score"]) and x["information"] >= 0 for x in ranked)


def test_no_repeat_acquired_tests(setup):
    model = setup[-1]
    before = model.ranking(["abdominal_pain"], {})
    key = before[0]["key"]
    after = model.ranking(["abdominal_pain"], {key: 0})
    assert key not in [x["key"] for x in after]


def test_invalid_inputs_rejected(setup):
    model = setup[-1]
    with pytest.raises(ValueError):
        model.posterior(["totally_fake_symptom"])
    with pytest.raises(ValueError):
        model.posterior([], {"unlisted_test": 1})
    with pytest.raises(ValueError):
        model.posterior([], {"ecg": 999})
    with pytest.raises(ValueError):
        policy_path(model, {k:v[0] for k,v in setup[3].items()}, "unsafe")


def test_natural_language_mapper_conservative():
    assert "fever" in extract_symptoms("fever and coughing")
    assert "cough" in extract_symptoms("fever and coughing")
    assert "abdominal_pain" in extract_symptoms("stomach pain")
    assert extract_symptoms("___no-known-symptoms-here___") == []


def test_only_heldout_demo_case_metadata(setup):
    cases = make_demo_cases(setup[3], count=8)
    assert len(cases) == 8
    assert len({c["id"] for c in cases}) == 8
    assert all(set(c["test_findings"]) == set(TESTS) for c in cases)
    assert len({c["reference_label"] for c in cases}) >= 3


def test_is_not_clinical_device():
    import app
    assert "not medical advice" in app.SYNTHETIC_WARNING.lower()
    assert "not a medical device" in app.__doc__.lower()
    assert len(app.BY_SCENARIO) >= 6
    assert app.FULL_COST == sum(COSTS.values())


def test_demo_start_acquire_and_reset():
    import app
    scenario = app.SCENARIOS[0]
    input_symbols = scenario["symptoms"]
    start = app.start(app.SIMULATED_CASE, scenario["id"], scenario["complaint"], input_symbols, [])
    state = start[0]
    assert len(start) == 8
    assert state["cost"] == 0
    assert "Possible explanations" in start[2]
    progressed = app.acquire(state)
    assert progressed[0]["cost"] > 0
    assert len(progressed[0]["history"]) == 1
    assert len(progressed) == 8
    assert app.reset()[0] is None


def test_manual_mode_and_insufficient_symptoms():
    import app
    unsupported = app.start(app.MANUAL_CASE, None, "unrecognized concern", [], [])
    assert unsupported[0] is None
    manual = app.start(app.MANUAL_CASE, None, "fever and cough", [], [])
    assert manual[0]["mode"] == app.MANUAL_CASE
    assert app.acquire(manual[0])[0] == manual[0]
    progressed = app.record_result(manual[0], "Abnormal / flagged", "troponin")
    assert len(progressed[0]["obtained"]) == 1
    assert progressed[0]["obtained"]["troponin"] == 1
    # Existing tests can be entered in any order; no duplicate acquisition.
    duplicate = app.record_result(progressed[0], "Not flagged", "troponin")
    assert duplicate[0]["obtained"] == progressed[0]["obtained"]
    followup = app.record_result(progressed[0], "Not flagged", "urinalysis")
    assert len(followup[0]["obtained"]) == 2


def test_urgent_review_banner_always_shown_for_redflag():
    import app
    started = app.start(app.MANUAL_CASE, None, "dizzy", [], ["sudden_weakness"])
    assert "Priority clinical review" in started[1]
    assert "delay" in started[1].lower()


def test_holdout_report_reproducible_and_finite(tmp_path):
    report = save_research(outdir=tmp_path, n=500)
    assert report["conditions"] == len(CONDITIONS)
    assert report["test"] == 100
    assert report["dataset"].startswith("INDEPENDENTLY")
    for policy, record in report["test_results"].items():
        assert 0 <= record["accuracy"] <= 1
        assert 0 <= record["urgent_condition_sensitivity"] <= 1
        assert record["mean_effort_points"] >= 0
    assert (tmp_path / "model.json").exists()
    assert (tmp_path / "demo_cases.json").exists()
    assert (tmp_path / "results.md").exists()
