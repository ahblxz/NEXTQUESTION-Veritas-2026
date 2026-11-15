"""NEXTQUESTION multi-condition diagnostic information acquisition research simulator.

All cases, positive-test rates, costs and labels in this module are SYNTHETIC.
No clinical accuracy, test avoidance, or financial savings are established.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEED = 2026
SYMPTOMS = {
    "fever": "Fever", "cough": "Cough", "breathlessness": "Shortness of breath",
    "chest_pain": "Chest pain", "palpitations": "Palpitations",
    "leg_swelling": "Leg swelling", "fatigue": "Fatigue",
    "dizziness": "Dizziness", "headache": "Headache",
    "weakness": "New weakness", "abdominal_pain": "Abdominal pain",
    "vomiting": "Nausea or vomiting", "diarrhea": "Diarrhea",
    "dysuria": "Painful urination", "frequency": "Frequent urination",
    "flank_pain": "Flank pain", "thirst": "Excessive thirst",
    "cold": "Cold sensitivity", "weight_change": "Unexplained weight change",
    "joint_pain": "Joint pain", "rash": "Rash",
}
TESTS = {
    "pulse_ox": ("Oxygen saturation", 1),
    "glucose": ("Blood glucose", 1),
    "cbc": ("Complete blood count", 3),
    "crp": ("Inflammation marker (CRP)", 3),
    "urinalysis": ("Urinalysis", 3),
    "ecg": ("ECG", 4),
    "electrolytes": ("Electrolyte panel", 4),
    "chest_xray": ("Chest X-ray", 6),
    "troponin": ("Troponin", 7),
    "bnp": ("BNP blood test", 7),
    "tsh": ("Thyroid function (TSH)", 5),
    "ddimer": ("D-dimer", 6),
    "abdominal_us": ("Abdominal ultrasound", 9),
    "head_ct": ("Head CT", 16),
    "spirometry": ("Spirometry", 5),
    "hemoglobin": ("Hemoglobin", 2),
}
# name, category, urgent, high-frequency symptoms, weaker associated symptoms,
# strongly positive findings, weaker associated abnormal findings, prevalence weight.
# The associations are deliberately stylized teaching assumptions, NOT actual
# sensitivities, specificities, prevalence estimates or test recommendations.
CONDITIONS = [
    ("Viral respiratory illness", "Respiratory", False, "fever cough fatigue", "headache", "crp", "cbc", 15),
    ("Pneumonia", "Respiratory", True, "fever cough breathlessness", "fatigue chest_pain", "chest_xray crp pulse_ox", "cbc", 7),
    ("Asthma-like episode", "Respiratory", False, "cough breathlessness", "chest_pain", "spirometry pulse_ox", "chest_xray", 8),
    ("Pulmonary embolism-like syndrome", "Respiratory", True, "breathlessness chest_pain", "leg_swelling palpitations", "ddimer pulse_ox", "troponin", 3),
    ("Heart failure-like syndrome", "Cardiovascular", True, "breathlessness leg_swelling fatigue", "palpitations", "bnp chest_xray", "ecg pulse_ox", 4),
    ("Arrhythmia-like syndrome", "Cardiovascular", True, "palpitations dizziness", "breathlessness chest_pain", "ecg", "electrolytes troponin", 5),
    ("Acute coronary syndrome-like", "Cardiovascular", True, "chest_pain breathlessness", "vomiting fatigue", "troponin ecg", "cbc", 3),
    ("Gastroenteritis-like", "Gastrointestinal", False, "diarrhea vomiting abdominal_pain", "fever", "electrolytes", "crp", 12),
    ("Appendicitis-like", "Gastrointestinal", True, "abdominal_pain vomiting fever", "diarrhea", "crp cbc abdominal_us", "electrolytes", 4),
    ("Gallbladder inflammation-like", "Gastrointestinal", True, "abdominal_pain vomiting", "fever", "abdominal_us crp", "cbc", 4),
    ("Lower urinary tract infection", "Urinary", False, "dysuria frequency", "fever abdominal_pain", "urinalysis", "cbc crp", 9),
    ("Kidney infection-like", "Urinary", True, "flank_pain dysuria fever", "vomiting frequency", "urinalysis crp", "cbc", 5),
    ("Kidney stone-like", "Urinary", False, "flank_pain abdominal_pain vomiting", "dysuria", "urinalysis abdominal_us", "crp", 5),
    ("Migraine-like headache", "Neurologic", False, "headache vomiting", "dizziness", "", "head_ct", 8),
    ("Stroke-like syndrome", "Neurologic", True, "weakness dizziness headache", "vomiting", "head_ct", "ecg", 3),
    ("Iron-deficiency-anemia-like", "General/Metabolic", False, "fatigue dizziness", "breathlessness", "hemoglobin cbc", "glucose", 8),
    ("High-glucose syndrome", "General/Metabolic", True, "thirst frequency fatigue", "weight_change vomiting", "glucose", "electrolytes urinalysis", 7),
    ("Thyroid-related syndrome", "General/Metabolic", False, "fatigue cold weight_change", "dizziness", "tsh", "cbc", 6),
    ("Dehydration-like syndrome", "General/Metabolic", False, "thirst dizziness fatigue", "diarrhea vomiting", "electrolytes", "glucose", 9),
    ("Inflammatory joint condition-like", "Musculoskeletal/Skin", False, "joint_pain fatigue", "fever rash", "crp", "cbc", 5),
    ("Allergic-reaction-like", "Musculoskeletal/Skin", True, "rash breathlessness", "vomiting", "pulse_ox", "cbc", 5),
]
RED_FLAGS = {
    "sudden_weakness": "Sudden weakness, facial droop, or speech changes",
    "severe_chest": "Severe or ongoing chest pain",
    "severe_breathing": "Severe difficulty breathing",
    "confusion": "New confusion or loss of consciousness",
    "uncontrolled_bleeding": "Uncontrolled bleeding",
    "severe_abdomen": "Severe or worsening abdominal pain",
}
KEYS = [x[0] for x in CONDITIONS]
URGENCY = np.array([bool(x[2]) for x in CONDITIONS])
COSTS = {k: v[1] for k, v in TESTS.items()}
TEST_KEYS = tuple(TESTS)
SYMPTOM_KEYS = tuple(SYMPTOMS)
SYNTHETIC_NOTE = ("SYNTHETIC research simulation: no real patients, clinical test "
                  "prices, medical validation, or proven savings. Not medical advice.")


def _split(text):
    return tuple(x for x in str(text).split() if x)


def generate_cases(n=4800, seed=SEED):
    """Generate illustrative multi-condition cases using documented assumptions."""
    rng = np.random.default_rng(seed)
    weights = np.asarray([row[7] for row in CONDITIONS], dtype=float)
    diagnosis = rng.choice(len(CONDITIONS), size=n, p=weights / weights.sum())
    s_prob = np.full((len(CONDITIONS), len(SYMPTOMS)), .06)
    t_prob = np.full((len(CONDITIONS), len(TESTS)), .07)
    # Syndromes intentionally share features and have imperfect/noisy test findings.
    for i, (_, _, _, core, associated, strong, weak, _) in enumerate(CONDITIONS):
        for name in _split(core):
            s_prob[i, SYMPTOM_KEYS.index(name)] = .72
        for name in _split(associated):
            j = SYMPTOM_KEYS.index(name)
            s_prob[i, j] = max(s_prob[i, j], .30)
        for name in _split(strong):
            t_prob[i, TEST_KEYS.index(name)] = .78
        for name in _split(weak):
            j = TEST_KEYS.index(name)
            t_prob[i, j] = max(t_prob[i, j], .34)
    symptoms = (rng.random((n, len(SYMPTOMS))) < s_prob[diagnosis]).astype(np.uint8)
    tests = (rng.random((n, len(TESTS))) < t_prob[diagnosis]).astype(np.uint8)
    return {"diagnosis": diagnosis, "symptoms": symptoms, "tests": tests,
            "case_id": np.arange(n, dtype=np.int64)}


def split_cases(data, seed=SEED):
    """Split *by case* before fitting any likelihoods (no cross-split reuse)."""
    rng = np.random.default_rng(seed + 2)
    order = rng.permutation(len(data["diagnosis"]))
    a, b = int(len(order) * .65), int(len(order) * .80)
    return tuple({k: arr[indices] for k, arr in data.items()}
                 for indices in (order[:a], order[a:b], order[b:]))


@dataclass
class BayesTestModel:
    prior: np.ndarray
    symptom_likelihood: np.ndarray
    test_likelihood: np.ndarray
    effort_exponent: float = .62

    @classmethod
    def fit(cls, data):
        y = data["diagnosis"]
        count = np.bincount(y, minlength=len(CONDITIONS))
        prior = (count + 1) / (len(y) + len(CONDITIONS))
        ps, pt = [], []
        for i in range(len(CONDITIONS)):
            ps.append((data["symptoms"][y == i].sum(axis=0) + 1) / (count[i] + 2))
            pt.append((data["tests"][y == i].sum(axis=0) + 1) / (count[i] + 2))
        return cls(prior, np.array(ps), np.array(pt))

    def posterior(self, present=(), obtained=None):
        present = set(present)
        if not present.issubset(SYMPTOMS):
            raise ValueError(f"Unknown symptoms: {present - set(SYMPTOMS)}")
        obtained = obtained or {}
        if not set(obtained).issubset(TESTS):
            raise ValueError("Unknown test key")
        if any(v not in (0, 1) for v in obtained.values()):
            raise ValueError("Test findings must be binary synthetic outcomes")
        logp = np.log(self.prior)
        # Unreported symptoms are UNKNOWN, never treated as clinically absent.
        for name in sorted(present):
            logp += np.log(self.symptom_likelihood[:, SYMPTOM_KEYS.index(name)])
        for name, value in obtained.items():
            p = self.test_likelihood[:, TEST_KEYS.index(name)]
            logp += np.log(p if value else (1 - p))
        logp -= np.max(logp)
        probability = np.exp(logp)
        return probability / probability.sum()

    def candidates(self, present=(), obtained=None, top=5):
        prob = self.posterior(present, obtained)
        indices = np.argsort(-prob)[:top]
        return [{"diagnosis": KEYS[i], "category": CONDITIONS[i][1],
                 "probability": float(prob[i]), "urgent": bool(URGENCY[i])}
                for i in indices]

    def ranking(self, present=(), obtained=None, budget=30):
        obtained = obtained or {}
        posterior = self.posterior(present, obtained)
        entropy = self._entropy(posterior)
        ranked = []
        for j, name in enumerate(TEST_KEYS):
            if name in obtained or COSTS[name] > budget:
                continue
            likelihood = self.test_likelihood[:, j]
            prob_positive = float(posterior @ likelihood)
            p1 = posterior * likelihood
            p0 = posterior * (1 - likelihood)
            p1 /= p1.sum()
            p0 /= p0.sum()
            gain = max(0., float(entropy - (prob_positive * self._entropy(p1)
                                             + (1 - prob_positive) * self._entropy(p0))))
            score = gain / (COSTS[name] ** self.effort_exponent)
            ranked.append({"key": name, "name": TESTS[name][0], "cost": COSTS[name],
                           "information": round(gain, 5), "score": round(score, 6),
                           "abnormal_likelihood": round(prob_positive, 3)})
        return sorted(ranked, key=lambda r: (-r["score"], r["cost"], r["name"]))

    @staticmethod
    def _entropy(p):
        p = np.asarray(p, dtype=float)
        return float(-np.sum(p * np.log(np.clip(p, 1e-12, 1))))


def policy_path(model, case, strategy="adaptive", max_tests=6, budget=22,
                confidence=.75):
    present = [name for i, name in enumerate(SYMPTOM_KEYS) if case["symptoms"][i]]
    ordered = None
    if strategy == "fixed":
        ordered = [x["key"] for x in model.ranking((), {}, budget=10000)]
    elif strategy == "cheapest":
        ordered = sorted(TESTS, key=lambda k: (COSTS[k], k))
    elif strategy not in ("adaptive", "all"):
        raise ValueError("Unsupported policy")
    obtained, path, cost = {}, [], 0
    remaining_budget = sum(COSTS.values()) if strategy == "all" else budget
    number = len(TESTS) if strategy == "all" else max_tests
    for step in range(number):
        p = model.posterior(present, obtained)
        if strategy != "all" and obtained and float(p.max()) >= confidence:
            break
        if strategy in ("fixed", "cheapest"):
            picks = [k for k in ordered if k not in obtained and COSTS[k] <= remaining_budget - cost]
            next_key = picks[0] if picks else None
        else:
            ranked = model.ranking(present, obtained, remaining_budget - cost)
            next_key = ranked[0]["key"] if ranked else None
        if next_key is None:
            break
        obtained[next_key] = int(case["tests"][TEST_KEYS.index(next_key)])
        cost += COSTS[next_key]
        path.append(next_key)
    p = model.posterior(present, obtained)
    return {"prediction": int(np.argmax(p)), "confidence": float(np.max(p)),
            "actual": int(case["diagnosis"]), "cost": cost, "tests": path,
            "count": len(path), "urgent_score": float(p[URGENCY].sum())}


def evaluate(model, data, strategy, max_tests=6, budget=22, confidence=.75):
    results = []
    for i in range(len(data["diagnosis"])):
        case = {k: v[i] for k, v in data.items()}
        results.append(policy_path(model, case, strategy, max_tests, budget, confidence))
    correct = np.array([r["prediction"] == r["actual"] for r in results])
    urgent = np.array([bool(URGENCY[r["actual"]]) for r in results])
    recognized = np.array([bool(URGENCY[r["prediction"]]) for r in results])
    return {
        "strategy": strategy, "n": len(results),
        "accuracy": float(correct.mean()),
        "urgent_condition_sensitivity": float((recognized & urgent).sum() / max(1, urgent.sum())),
        "urgent_conditions_in_test": int(urgent.sum()),
        "mean_tests": float(np.mean([r["count"] for r in results])),
        "mean_effort_points": float(np.mean([r["cost"] for r in results])),
        "all_case_costs": [r["cost"] for r in results],
        "all_case_correct": [int(c) for c in correct],
    }


ALIASES = {
    "fever": ("fever", "temperature", "hot and shivery"),
    "cough": ("cough", "coughing"),
    "breathlessness": ("shortness of breath", "breathless", "difficulty breathing", "cant breathe"),
    "chest_pain": ("chest pain", "chest hurts", "chest pressure"),
    "palpitations": ("palpitations", "racing heart", "heart pounding"),
    "leg_swelling": ("leg swelling", "swollen legs", "ankle swelling"),
    "fatigue": ("fatigue", "tired", "exhausted"),
    "dizziness": ("dizziness", "dizzy", "lightheaded"),
    "headache": ("headache", "head pain"),
    "weakness": ("weakness", "weak arm", "weak leg"),
    "abdominal_pain": ("abdominal pain", "stomach pain", "belly pain", "tummy pain"),
    "vomiting": ("vomiting", "throwing up", "nausea", "sick to stomach"),
    "diarrhea": ("diarrhea", "diarrhoea", "loose stools"),
    "dysuria": ("painful urination", "burning pee", "burning urination"),
    "frequency": ("frequent urination", "peeing often", "urinating often"),
    "flank_pain": ("flank pain", "side pain", "kidney pain"),
    "thirst": ("thirst", "very thirsty"),
    "cold": ("cold intolerance", "feeling cold", "cold sensitivity"),
    "weight_change": ("weight loss", "weight gain", "weight changes"),
    "joint_pain": ("joint pain", "painful joints"),
    "rash": ("rash", "hives"),
}


def extract_symptoms(text):
    """Conservative keyword mapper, not an LLM medical language interpreter."""
    value = " ".join(str(text).lower().replace("-", " ").split())
    return [key for key, tokens in ALIASES.items()
            if any(token in value for token in tokens)]


def make_demo_cases(data, count=12):
    """Select only validation-split synthetic cases, NEVER the locked test split."""
    chosen, seen = [], set()
    for i in range(len(data["diagnosis"])):
        category = CONDITIONS[int(data["diagnosis"][i])][1]
        if category not in seen:
            chosen.append(i)
            seen.add(category)
    for i in range(len(data["diagnosis"])):
        if len(chosen) >= count:
            break
        if i not in chosen:
            chosen.append(i)
    cases = []
    for idx, i in enumerate(chosen[:count], 1):
        symptoms = [key for j, key in enumerate(SYMPTOM_KEYS) if data["symptoms"][i, j]]
        cases.append({
            "id": f"Scenario {idx:02d}",
            "complaint": ", ".join(SYMPTOMS[x].lower() for x in symptoms[:4]),
            "symptoms": symptoms,
            "test_findings": {key: int(data["tests"][i, j]) for j, key in enumerate(TEST_KEYS)},
            "reference_label": KEYS[int(data["diagnosis"][i])],
        })
    return cases


def save_research(outdir="artifacts/hospital", n=4800, seed=SEED):
    """Deterministic train/validation/test with locked held-out evaluation."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    records = generate_cases(n, seed)
    train, val, test = split_cases(records, seed)
    model = BayesTestModel.fit(train)
    # Select the cost penalty AND the stopping threshold using validation only.
    # This is a prespecified grid search, not hand-tuning on held-out outcomes.
    candidates = []
    for exponent in (.62, .95, 1.25):
        model.effort_exponent = exponent
        for threshold in (.60, .70, .80):
            adaptive_val = evaluate(model, val, "adaptive", confidence=threshold)
            fixed_val = evaluate(model, val, "fixed", confidence=threshold)
            # Prefer policies improving accuracy and effort relative to fixed.
            dominates = (adaptive_val["accuracy"] >= fixed_val["accuracy"] and
                         adaptive_val["mean_effort_points"] <= fixed_val["mean_effort_points"])
            utility = (adaptive_val["accuracy"] - .018 * adaptive_val["mean_effort_points"])
            candidates.append((dominates, utility, -adaptive_val["mean_effort_points"],
                               exponent, threshold))
    best = max(candidates)
    model.effort_exponent, confidence = best[3], best[4]
    strategies = {k: evaluate(model, test, k, confidence=confidence)
                  for k in ("adaptive", "fixed", "cheapest", "all")}
    # Aggregate only; never imply counterfactual benefit to any real patient.
    clean = {k: {p: val for p, val in v.items()
                 if p not in ("all_case_costs", "all_case_correct")}
             for k, v in strategies.items()}
    rng = np.random.default_rng(seed + 90)
    a, b = strategies["adaptive"], strategies["fixed"]
    costdiff = np.asarray(a["all_case_costs"]) - np.asarray(b["all_case_costs"])
    accdiff = np.asarray(a["all_case_correct"]) - np.asarray(b["all_case_correct"])
    draws = rng.integers(0, len(costdiff), size=(1200, len(costdiff)))
    intervals = {
        "adaptive_minus_fixed_cost_ci95": np.quantile(costdiff[draws].mean(axis=1), [.025, .975]).tolist(),
        "adaptive_minus_fixed_accuracy_ci95": np.quantile(accdiff[draws].mean(axis=1), [.025, .975]).tolist(),
    }
    payload = {
        "project": "NEXTQUESTION · Multi-condition diagnostic test prioritization",
        "dataset": "INDEPENDENTLY GENERATED SYNTHETIC SCENARIOS — not Synthea patients or real EHR",
        "generating_assumptions": "Manually specified stylized symptom/test likelihoods; independent conditional on diagnosis; results are not clinical sensitivities",
        "n": n, "train": len(train["diagnosis"]), "validation": len(val["diagnosis"]),
        "test": len(test["diagnosis"]), "conditions": len(CONDITIONS),
        "clinical_categories": sorted(set(row[1] for row in CONDITIONS)),
        "test_types": len(TESTS), "seed": seed, "confidence_from_validation": confidence,
        "effort_exponent_from_validation": model.effort_exponent,
        "units": "Synthetic effort points, NOT AED, prices, medical necessity or savings",
        "test_results": clean, "paired_bootstrap_ci": intervals,
        "warning": SYNTHETIC_NOTE,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    parameters = {"prior": model.prior.tolist(),
                  "symptom_likelihood": model.symptom_likelihood.tolist(),
                  "test_likelihood": model.test_likelihood.tolist(),
                  "effort_exponent": model.effort_exponent, "threshold": confidence}
    (out / "model.json").write_text(json.dumps(parameters), encoding="utf-8")
    # The evaluator must NEVER expose final held-out cases via the demo.
    assert not set(val["case_id"]).intersection(test["case_id"])
    assert not set(train["case_id"]).intersection(test["case_id"])
    cases = make_demo_cases(val)
    demo_source_ids = set(int(x) for x in val["case_id"][:len(val)])
    assert demo_source_ids.isdisjoint(set(int(x) for x in test["case_id"]))
    payload["demo_source"] = "validation_only"
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "demo_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    lines = [
        "# NEXTQUESTION — synthetic multi-condition research results", "",
        "**Research prototype, not medical advice. No real patient data, prices, observed cost savings, or clinical validation.**",
        "", f"Deterministic synthetic generator: {n} cases, "
        f"{len(CONDITIONS)} conditions across {len(payload['clinical_categories'])} categories and {len(TESTS)} test types.",
        f"Case-level split: {payload['train']} train / {payload['validation']} validation / {payload['test']} held-out.",
        f"Stopping threshold {confidence:.2f} selected using validation only.", "",
        "| Policy | Synthetic diagnostic accuracy | Urgent-class recognition | Mean tests | Mean effort points |",
        "|---|---:|---:|---:|---:|"
    ]
    for key in ("adaptive", "fixed", "cheapest", "all"):
        z = clean[key]
        lines.append(f"| {key} | {z['accuracy']:.1%} | {z['urgent_condition_sensitivity']:.1%} | "
                     f"{z['mean_tests']:.2f} | {z['mean_effort_points']:.2f} |")
    lines += [
        "", f"Adaptive − fixed mean effort 95% paired bootstrap CI: {intervals['adaptive_minus_fixed_cost_ci95']}",
        f"Adaptive − fixed accuracy 95% paired bootstrap CI: {intervals['adaptive_minus_fixed_accuracy_ci95']}",
        "",
        "## Critical limitations", "",
        "- This is a simulation using manually specified condition-specific test signals, not real medical diagnoses.",
        "- The model may miss emergencies; urgency sensitivity here is a synthetic label-recognition metric, not triage safety.",
        "- All tests are reduced to artificial abnormal/not-flagged findings; this is not clinical workflow.",
        "- Unreported symptoms are unknown, not ruled out. Free-text matching is conservative keyword matching.",
        "- Cost figures are synthetic effort points; no evidence of fewer real tests or reduced bills.",
        "- Do not use for patients, test ordering, emergencies or postponing medical care.",
        "- Cases outside the simulated condition catalog are unsupported; clinician judgment is required.",
        "",
        "## Next research step", "",
        "Evaluate prospectively on authorized real deidentified encounters and actual test prices "
        "with licensed clinicians and safety oversight before making clinical claims.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_model(path="artifacts/hospital/model.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BayesTestModel(
        np.asarray(data["prior"], dtype=float),
        np.asarray(data["symptom_likelihood"], dtype=float),
        np.asarray(data["test_likelihood"], dtype=float),
        float(data.get("effort_exponent", .62)),
    ), float(data["threshold"])
