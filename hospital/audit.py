"""Audit of locked synthetic holdout, error types and out-of-distribution stress tests.

IMPORTANT: This is not medical validation. The original generator is entirely
artificial and its clinical/test associations are teaching assumptions.
This module does NOT tune hyperparameters, fit on evaluation cases, or reveal
holdout rows to the interactive demonstration.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hospital import engine


def _urgent_binary_counts(actual, predicted):
    """Synthetic urgent-class recognition, not validated emergency triage."""
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    n = int(len(actual))
    assert tp + tn + fp + fn == n
    return {
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "n": n, "positive_support": int(actual.sum()),
        "negative_support": int((~actual).sum()),
        "confusion_matrix_rows_actual_nonurgent_urgent": [[tn, fp], [fn, tp]],
    }


def _multiclass_diagnostics(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    n = len(engine.CONDITIONS)
    confusion = np.zeros((n, n), dtype=int)
    np.add.at(confusion, (actual, predicted), 1)
    per_condition = []
    for i, row in enumerate(engine.CONDITIONS):
        tp = int(confusion[i, i])
        fn = int(confusion[i, :].sum()) - tp
        fp = int(confusion[:, i].sum()) - tp
        tn = len(actual) - tp - fn - fp
        per_condition.append({
            "condition": row[0], "category": row[1], "synthetic_urgent": row[2],
            "true_positive": tp, "false_negative": fn,
            "false_positive": fp, "true_negative": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "support": int(confusion[i, :].sum())
        })
    return {
        "labels": engine.KEYS,
        "actual_rows_by_predicted_columns": confusion.tolist(),
        "per_condition_one_vs_rest": per_condition,
        "macro_precision": float(np.mean([c["precision"] for c in per_condition])),
        "macro_recall": float(np.mean([c["recall"] for c in per_condition])),
    }


def individual_paths(model, data, policy, confidence):
    return [
        engine.policy_path(
            model, {key: val[i] for key, val in data.items()},
            strategy=policy, confidence=confidence
        )
        for i in range(len(data["diagnosis"]))
    ]


def _one_failure(paths, dataset, prefer_urgent_miss=True):
    urgent_misses = [
        i for i, row in enumerate(paths)
        if engine.URGENCY[row["actual"]] and not engine.URGENCY[row["prediction"]]
    ]
    errors = [i for i, row in enumerate(paths) if row["actual"] != row["prediction"]]
    if not errors:
        return None
    idx = urgent_misses[0] if (urgent_misses and prefer_urgent_miss) else errors[0]
    row = paths[idx]
    symptoms = [
        engine.SYMPTOMS[name] for j, name in enumerate(engine.SYMPTOM_KEYS)
        if dataset["symptoms"][idx, j]
    ]
    acquired = [
        {"test": engine.TESTS[key][0],
         "artificial_finding": "flagged" if dataset["tests"][idx, engine.TEST_KEYS.index(key)] else "not flagged",
         "effort_points": engine.COSTS[key]}
        for key in row["tests"]
    ]
    return {
        "synthetic_case_id": int(dataset["case_id"][idx]),
        "actual_simulated_condition": engine.KEYS[row["actual"]],
        "predicted_simulated_condition": engine.KEYS[row["prediction"]],
        "actual_urgent_label": bool(engine.URGENCY[row["actual"]]),
        "predicted_urgent_label": bool(engine.URGENCY[row["prediction"]]),
        "reported_synthetic_symptoms": symptoms,
        "disclosed_artificial_test_findings": acquired,
        "simulated_effort_points": row["cost"],
        "model_top_probability": round(row["confidence"], 5),
        "possible_reason_not_proven": (
            "Overlap between the hand-designed symptom patterns, stochastic "
            "binary test findings, conditional-independence assumptions, and "
            "the limited simulated testing budget. Cannot assign causality "
            "for an individual artificial error."
        ),
    }


def _stress_dataset(seed, mode):
    """Independently generated synthetic cases; fixed model, no retuning."""
    original = engine.generate_cases(n=1550, seed=seed)
    rng = np.random.default_rng(seed + 533)
    data = {k: v.copy() for k, v in original.items()}
    if mode == "missing_30pct_positive_symptoms":
        present = data["symptoms"].astype(bool)
        missing = rng.random(present.shape) < .30
        data["symptoms"][present & missing] = 0  # unreported, NOT known absent
    elif mode == "flipped_10pct_test_indicators":
        flips = rng.random(data["tests"].shape) < .10
        data["tests"] ^= flips.astype(np.uint8)
    elif mode == "urgent_prevalence_shift":
        urgent = engine.URGENCY[data["diagnosis"]]
        keep = rng.random(len(urgent)) < np.where(urgent, .95, .40)
        data = {k: v[keep] for k, v in data.items()}
    elif mode in ("reference_new_seed", "cost_shift"):
        pass
    else:
        raise ValueError(mode)
    assert len(data["diagnosis"]) > 200
    return data


def _stress_evaluate(model, confidence):
    # No test or shift scenarios influence the already-selected model parameters.
    modes = (
        "reference_new_seed", "missing_30pct_positive_symptoms",
        "flipped_10pct_test_indicators", "urgent_prevalence_shift",
        "cost_shift"
    )
    report = {}
    for j, mode in enumerate(modes):
        data = _stress_dataset(89431 + 101 * j, mode)
        if mode == "cost_shift":
            # Artificial counterfactual: expensive laboratory/imaging tests
            # cost twice as much; cheap investigations retain baseline weights.
            altered = {
                name: int(cost * (2 if cost >= 5 else 1))
                for name, cost in engine.COSTS.items()
            }
            with patch.dict(engine.COSTS, altered, clear=True):
                adaptive = engine.evaluate(model, data, "adaptive", confidence=confidence)
                fixed = engine.evaluate(model, data, "fixed", confidence=confidence)
        else:
            adaptive = engine.evaluate(model, data, "adaptive", confidence=confidence)
            fixed = engine.evaluate(model, data, "fixed", confidence=confidence)
        report[mode] = {
            "n": len(data["diagnosis"]),
            "adaptive_accuracy": adaptive["accuracy"],
            "fixed_accuracy": fixed["accuracy"],
            "adaptive_mean_tests": adaptive["mean_tests"],
            "fixed_mean_tests": fixed["mean_tests"],
            "adaptive_simulated_effort": adaptive["mean_effort_points"],
            "fixed_simulated_effort": fixed["mean_effort_points"],
            "assumptions": {
                "reference_new_seed": "Same synthetic generating assumptions; new random seed",
                "missing_30pct_positive_symptoms": "Randomly suppress 30% of otherwise reported positive symptoms",
                "flipped_10pct_test_indicators": "Randomly flip 10% of latent artificial test bits",
                "urgent_prevalence_shift": "Retain urgent-class cases more often; independent simulated generation",
                "cost_shift": "Double the synthetic cost of every test costing at least 5 points",
            }[mode],
            "disclaimer": "Artificial stress scenario; not clinical external validation",
        }
    return report


def audit(outdir="artifacts/hospital", n=4800, seed=engine.SEED):
    """Only evaluate locked holdout after the main training stage chooses settings."""
    out = Path(outdir)
    summary = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    model, threshold = engine.load_model(str(out / "model.json"))
    assert summary["seed"] == seed and summary["n"] == n
    assert summary["confidence_from_validation"] == threshold
    assert summary["effort_exponent_from_validation"] == model.effort_exponent
    data = engine.generate_cases(n, seed)
    train, validation, test = engine.split_cases(data, seed)
    train_ids = set(train["case_id"].tolist())
    val_ids = set(validation["case_id"].tolist())
    test_ids = set(test["case_id"].tolist())
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)

    # Model trained only on training split. Refitting exactly must reproduce every weight.
    reproducible = engine.BayesTestModel.fit(train)
    for key in ("prior", "symptom_likelihood", "test_likelihood"):
        assert np.array_equal(getattr(reproducible, key), getattr(model, key)), key

    demo = json.loads((out / "demo_cases.json").read_text(encoding="utf-8"))
    correct_val = engine.make_demo_cases(validation)
    assert demo == correct_val, "Demo records MUST be exclusively validation examples"
    assert summary.get("demo_source") == "validation_only"
    paths = {
        policy: individual_paths(model, test, policy, threshold)
        for policy in ("adaptive", "fixed", "cheapest", "all")
    }
    report = {
        "scope": "Held-out artificial cases; not patient or test safety validation",
        "source": "Hand-designed synthetic generator in hospital/engine.py",
        "splits_disjoint_by_case_id": True,
        "demo_contains_only_validation_cases": True,
        "model_refit_equal_training_only": True,
        "hyperparameters_locked_before_holdout": {
            "confidence": threshold,
            "effort_exponent": model.effort_exponent,
        },
        "synthetic_n": len(test["diagnosis"]),
        "by_policy": {},
        "failure_examples": [],
    }
    for policy, rows in paths.items():
        actual = np.array([r["actual"] for r in rows], dtype=int)
        predicted = np.array([r["prediction"] for r in rows], dtype=int)
        actual_urgent = engine.URGENCY[actual]
        pred_urgent = engine.URGENCY[predicted]
        report["by_policy"][policy] = {
            "accuracy": float(np.mean(actual == predicted)),
            "urgent_vs_nonurgent": _urgent_binary_counts(actual_urgent, pred_urgent),
            "class_diagnostics": _multiclass_diagnostics(actual, predicted),
            "mean_tests": float(np.mean([r["count"] for r in rows])),
            "mean_effort_points": float(np.mean([r["cost"] for r in rows])),
        }
    report["failure_examples"] = [
        x for x in [
            _one_failure(paths["adaptive"], test, prefer_urgent_miss=True),
            _one_failure(paths["fixed"], test, prefer_urgent_miss=False),
        ] if x is not None
    ]
    report["stress_scenarios"] = _stress_evaluate(model, threshold)
    (out / "audit_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# NEXTQUESTION — transparent synthetic test evaluation",
        "",
        "**Not clinically validated.** These are fictional patients, fabricated binary test signals "
        "and hypothetical effort points, not hospital bills or real patient outcomes.",
        "",
        f"Training/validation/held-out: {summary['train']}/{summary['validation']}/{summary['test']}; "
        "demo **validation only**; no hidden test result lookahead.",
        f"Validation-selected settings: stop confidence {threshold:.2f}; "
        f"effort exponent {model.effort_exponent:.2f}.", "",
        "## Held-out comparisons", "",
        "| Strategy | Synthetic accuracy | Tests/case | Effort pts/case | Urgent TP | Urgent FP | Urgent FN | Urgent TN | Urgent precision | Urgent recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for name, d in report["by_policy"].items():
        b = d["urgent_vs_nonurgent"]
        lines.append(
            f"| {name} | {d['accuracy']:.1%} | {d['mean_tests']:.2f} | "
            f"{d['mean_effort_points']:.2f} | {b['true_positive']} | {b['false_positive']} | "
            f"{b['false_negative']} | {b['true_negative']} | {b['precision']:.1%} | {b['recall']:.1%} |"
        )
    lines += [
        "", "A false negative above means a **synthetic urgent-labeled condition** was "
        "classified as a non-urgent-labeled condition. This is not an emergency-triage assessment.",
        "The 2×2 matrices use rows = actual [non-urgent, urgent] and columns = predicted [non-urgent, urgent].",
    ]
    for name, d in report["by_policy"].items():
        lines.append(f"- {name} urgent-label matrix: {d['urgent_vs_nonurgent']['confusion_matrix_rows_actual_nonurgent_urgent']}")
    lines += ["", "## Detailed synthetic classification errors", "",
              "The machine-readable audit_metrics.json also contains all 21 one-vs-rest "
              "false-positive / false-negative counts and the complete 21×21 confusion "
              "matrix (actual rows, predicted columns)."]
    lines += ["", "## Genuine held-out simulation failure example", ""]
    if report["failure_examples"]:
        e = report["failure_examples"][0]
        lines += [
            f"- Artificial held-out case ID: {e['synthetic_case_id']}; true simulated syndrome: "
            f"**{e['actual_simulated_condition']}**; predicted **{e['predicted_simulated_condition']}**.",
            f"- Actual urgent label: {e['actual_urgent_label']}; predicted urgent label: {e['predicted_urgent_label']}.",
            f"- Reported symptoms: {', '.join(e['reported_synthetic_symptoms']) or '(none)'}",
            "- Disclosed findings: " + "; ".join(
                f"{step['test']}={step['artificial_finding']}"
                for step in e["disclosed_artificial_test_findings"]
            ),
            f"- {e['simulated_effort_points']} effort points, leading posterior weight {e['model_top_probability']:.1%}.",
            f"- Possible mechanism (not proven): {e['possible_reason_not_proven']}",
        ]
    lines += ["", "## Robustness (independent synthetic cohorts; no retuning)", "",
              "| Scenario | N | Adaptive accuracy | Fixed accuracy | Adaptive tests | Fixed tests | Adaptive effort | Fixed effort |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, d in report["stress_scenarios"].items():
        lines.append(
            f"| {mode} | {d['n']} | {d['adaptive_accuracy']:.1%} | "
            f"{d['fixed_accuracy']:.1%} | {d['adaptive_mean_tests']:.2f} | "
            f"{d['fixed_mean_tests']:.2f} | {d['adaptive_simulated_effort']:.2f} | "
            f"{d['fixed_simulated_effort']:.2f} |"
        )
    lines += [
        "", "## Interpretation and limits", "",
        "These stress tests vary aspects of the **same hand-designed generator**; "
        "they do not establish external clinical generalization. "
        "The system only ranks information within its limited artificial catalog.",
        "The full panel is an artificial comparator, not standard medical practice. "
        "There is no evidence of avoided real testing, lower medical bills, earlier diagnosis or lives saved.",
        "**This project is a prototype and is not medical advice.**"
    ]
    (out / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = audit()
    a = result["by_policy"]["adaptive"]["urgent_vs_nonurgent"]
    print("Synthetic urgent-label matrix (TN FP / FN TP):",
          a["confusion_matrix_rows_actual_nonurgent_urgent"])
    print(f"False positives={a['false_positive']} false negatives={a['false_negative']} "
          f"precision={a['precision']:.1%} recall={a['recall']:.1%}")
    print("Report:", Path("artifacts/hospital/audit_report.md").resolve())
