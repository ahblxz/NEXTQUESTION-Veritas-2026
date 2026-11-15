"""Independent, PUBLIC symptom-evidence benchmark.

Public source: https://github.com/hemasriram111/Disease-Symptom-Dataset
Pinned CSV commit: b0cda0aca4bd107f1e32bb58b0a0043aa9caf6ad
The dataset contains symptom lists and labels, *NOT laboratory test results*.
It is a secondary evaluation of information acquisition, NEVER proof of
diagnostic-test safety or clinical price savings.

Crucially: 4,920 source rows collapse into about 304 unique patterns.
We split unique symptom patterns BEFORE training, preventing duplicate leakage.
Do not redistribute the source CSV: its precise data license is not documented.
"""
import csv
import hashlib
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCE = (
    "https://raw.githubusercontent.com/hemasriram111/Disease-Symptom-Dataset/"
    "b0cda0aca4bd107f1e32bb58b0a0043aa9caf6ad/dataset.csv"
)


def load_public_dataset(url=SOURCE):
    with urllib.request.urlopen(url, timeout=35) as response:
        payload = response.read()
    csv_text = payload.decode("utf-8-sig")
    raw = list(csv.DictReader(io.StringIO(csv_text)))
    if len(raw) < 1000:
        raise ValueError("Unexpectedly small public benchmark")
    grouped = {}
    for row in raw:
        label = str(row.get("Disease", "")).strip()
        symptoms = tuple(sorted(set(
            str(val).strip().replace(" ", "_").lower()
            for key, val in row.items()
            if key.startswith("Symptom_") and val and str(val).strip()
        )))
        if not label or not symptoms:
            continue
        if symptoms in grouped and grouped[symptoms] != label:
            raise ValueError("Identical symptom pattern has conflicting labels")
        grouped[symptoms] = label
    return list(grouped.items()), {
        "source": SOURCE, "sha256": hashlib.sha256(payload).hexdigest(),
        "original_rows": len(raw), "unique_symptom_patterns": len(grouped),
        "scope": "Public disease/symptom table; no diagnostic laboratory/imaging results",
        "license_note": "Upstream license not clearly declared; do not redistribute source CSV",
    }


def stratified_pattern_split(rows, seed=241):
    by_class = defaultdict(list)
    for symptoms, label in rows:
        by_class[label].append((symptoms, label))
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in sorted(by_class):
        examples = by_class[label]
        if len(examples) < 4:
            raise ValueError(f"Class {label} has insufficient deduplicated patterns")
        order = rng.permutation(len(examples))
        test.append(examples[int(order[0])])
        val.append(examples[int(order[1])])
        train += [examples[int(j)] for j in order[2:]]
    fingerprints = [
        set(symptoms for symptoms, _ in split) for split in (train, val, test)
    ]
    assert not (fingerprints[0] & fingerprints[1] or
                fingerprints[1] & fingerprints[2] or
                fingerprints[0] & fingerprints[2])
    return train, val, test


class SymptomModel:
    def __init__(self, train):
        self.labels = tuple(sorted({label for _, label in train}))
        self.vocabulary = tuple(sorted({symptom for symptoms, _ in train for symptom in symptoms}))
        idx = {x: i for i, x in enumerate(self.vocabulary)}
        self.index = idx
        counts = np.zeros(len(self.labels), dtype=float)
        positive = np.zeros((len(self.labels), len(self.vocabulary)), dtype=float)
        for symptoms, label in train:
            j = self.labels.index(label)
            counts[j] += 1
            for symptom in symptoms:
                positive[j, idx[symptom]] += 1
        self.prior = (counts + 1) / (counts.sum() + len(self.labels))
        self.theta = (positive + 1) / (counts[:, None] + 2)

    def posterior(self, reported, answered):
        probs = np.log(self.prior)
        for symptom in set(reported):
            if symptom in self.index:
                probs += np.log(self.theta[:, self.index[symptom]])
        for symptom, response in answered.items():
            if symptom not in self.index:
                continue
            p = self.theta[:, self.index[symptom]]
            probs += np.log(p if response else 1 - p)
        probs -= probs.max()
        probs = np.exp(probs)
        return probs / probs.sum()

    def recommend(self, reported, answered, policy="adaptive"):
        offered = [
            name for name in self.vocabulary
            if name not in answered and name not in reported
        ]
        if not offered:
            return None
        if policy == "fixed":
            return offered[0]
        if policy != "adaptive":
            raise ValueError(policy)
        p = self.posterior(reported, answered)
        entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1)))
        best = (-float("inf"), "")
        for name in offered:
            likelihood = self.theta[:, self.index[name]]
            chance = float(p @ likelihood)
            pos = p * likelihood
            neg = p * (1 - likelihood)
            pos /= pos.sum()
            neg /= neg.sum()
            expected = (
                chance * (-np.sum(pos * np.log(np.clip(pos, 1e-12, 1))))
                + (1 - chance) * (-np.sum(neg * np.log(np.clip(neg, 1e-12, 1))))
            )
            gain = float(entropy - expected)
            if gain > best[0]:
                best = (gain, name)
        return best[1]


def evaluation(model, split, policy, confidence=0.80, max_questions=8):
    correct, asked, predictions = [], [], []
    for symptoms, actual in split:
        # Two reported symptoms are the initial observed evidence.
        reported = list(symptoms[:2])
        answered = {}
        for _ in range(max_questions):
            p = model.posterior(reported, answered)
            if answered and max(p) >= confidence:
                break
            question = model.recommend(reported, answered, policy)
            if question is None:
                break
            # The public table encodes *present symptoms*, not laboratory tests.
            answered[question] = int(question in symptoms)
        p = model.posterior(reported, answered)
        prediction = model.labels[int(np.argmax(p))]
        correct.append(int(prediction == actual))
        asked.append(len(answered))
        predictions.append(prediction)
    return {
        "accuracy": float(np.mean(correct)), "mean_symptom_questions": float(np.mean(asked)),
        "n": len(split), "errors": int(len(split) - sum(correct)),
        "predictions": predictions,
    }


def main(outdir="artifacts/hospital/public"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_public_dataset()
    train, validation, test = stratified_pattern_split(rows)
    model = SymptomModel(train)
    # A tiny, declared selection grid, validation-only; test labels are not consulted.
    candidates = []
    for confidence in (0.60, 0.80, 0.95):
        report = evaluation(model, validation, "adaptive", confidence)
        score = report["accuracy"] - 0.005 * report["mean_symptom_questions"]
        candidates.append((score, -report["mean_symptom_questions"], confidence))
    threshold = max(candidates)[2]
    adaptive = evaluation(model, test, "adaptive", threshold)
    fixed = evaluation(model, test, "fixed", threshold)
    prediction_outputs = {
        "adaptive": {k: v for k, v in adaptive.items() if k != "predictions"},
        "fixed": {k: v for k, v in fixed.items() if k != "predictions"},
    }
    report = {
        "source": provenance, "split": {
            "train_unique_patterns": len(train),
            "validation_unique_patterns": len(validation),
            "test_unique_patterns": len(test),
            "all_classes_in_each_split": True,
            "duplicate_pattern_leakage": False,
        },
        "model": "Simple naïve Bayes over public symptom observations only",
        "selected_confidence_validation": threshold,
        "test": prediction_outputs,
        "warning": (
            "Secondary public-data benchmark covers symptom questions only. "
            "No actual diagnostic blood tests, imaging, hospital bills or "
            "clinical necessity labels exist in this source. "
            "The dataset has extensive duplicates; deduplicate before splitting. "
            "The dataset's patient provenance is not clinically established."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Independent public symptom benchmark\n\n"
        f"Source: {SOURCE}\n\n"
        f"Original CSV rows: {provenance['original_rows']}; "
        f"unique patterns: {provenance['unique_symptom_patterns']}; "
        f"SHA-256: {provenance['sha256']}.\n\n"
        f"After de-duplication: {len(train)} training, {len(validation)} validation, "
        f"{len(test)} held-out patterns.\n\n"
        f"Adaptive symptom question accuracy {adaptive['accuracy']:.1%}, "
        f"mean questions {adaptive['mean_symptom_questions']:.2f}; "
        f"fixed {fixed['accuracy']:.1%}, mean questions "
        f"{fixed['mean_symptom_questions']:.2f}.\n\n"
        "**NOT a laboratory test-selection evaluation.** Public data are "
        "disease/symptom mappings, not clinical encounters. License not clearly "
        "documented upstream; no copy redistributed. No clinical benefit proven.\n",
        encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
