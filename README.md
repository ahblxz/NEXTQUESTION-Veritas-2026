# NEXTQUESTION
### Which diagnostic information is worth gathering next?

**AI + Health · Veritas AI International High School Hackathon 2026**

NEXTQUESTION is a clinician-oriented **research demonstration**, not a medical device. It explores adaptive information acquisition across different presenting complaints, rather than predicting only one condition. At each step a probabilistic model evaluates how much an unobserved finding is expected to reduce uncertainty and updates its ranking after a result is revealed.

**Two experiments are deliberately kept separate.**

| Research component | Source | What it can establish |
|---|---|---|
| Main interactive methodology demo | 4,800 self-generated fictional cases, 21 artificial syndrome labels, 16 binary test indicators | Reproducible, proof-of-mechanism comparison of adaptive versus fixed-order **simulated** testing |
| Public hospital-record pilot | Openly published, de-identified MIMIC-IV and MIMIC-IV-ED Demo 2.2 | Preliminary retrospective evaluation using **actually recorded** laboratory results; not a clinical test-ordering validation |
| Public symptom-only benchmark | Published Disease-Symptom Dataset, deduplicated by unique symptom pattern | Secondary symptom-question acquisition evaluation; **not** laboratory-test selection |

**This project is a prototype and is not medical advice.** Do not enter real patient-identifiable information. Never use it to diagnose an individual, order or omit actual medical tests, or delay urgent care. Neither experiment proves actual hospital cost savings.

## Open and reproduce the demonstration

[Run NEXTQUESTION in Google Colab](https://colab.research.google.com/github/ahblxz/NEXTQUESTION-Veritas-2026/blob/main/notebooks/02_run_complete_project.ipynb): choose **Runtime → Run all**. The final cell creates a temporary Gradio link. Keep the notebook session active.

For a local run, use Python 3.11 or newer:

    python -m pip install -r requirements.txt
    python -m pytest -q tests/test_hospital.py tests/test_audit.py
    python hospital_train.py --out artifacts/hospital --n 4800
    python -m hospital.audit
    python -m hospital.public_benchmark
    python -m hospital.mimic_real
    python -m pytest -q tests/test_real_data.py
    python app.py

The public MIMIC data download requires internet access. **The app never substitutes fake data when that download fails**: the separate hospital-record pilot displays setup instructions instead.

To build the web demo as a container:

    docker build -t nextquestion .
    docker run --rm -p 7860:7860 nextquestion

Open http://localhost:7860. Docker builds and preserves both research models and derived validation examples, but excludes all downloaded raw EHR tables. This is a **deployable educational application**, not an always-on publicly hosted service or approved clinical system.

## Main synthetic-method experiment (960 held-out artificial cases)

The model uses Bayesian updates from reported fictional symptoms and acquired artificial test results. For each candidate test, the selector computes expected entropy reduction over possible outcomes, adjusting for illustrative effort. It does not see unrevealed results. Stopping and effort-exponent hyperparameters were selected on validation only.

| Policy | Classification accuracy | Tests per simulated case | Arbitrary effort pts | Urgent-label FN | Urgent-label FP |
|---|---:|---:|---:|---:|---:|
| **Adaptive NEXTQUESTION** | **73.6%** | **2.79** | **7.31** | **74** | 71 |
| Fixed-order comparison | 70.0% | 3.53 | 7.31 | 94 | 69 |
| Cheapest-first comparison | 67.2% | 3.66 | 7.08 | 94 | 73 |
| Full hypothetical panel | 76.9% | 16.00 | 82.00 | 63 | 58 |

This is **21% fewer artificial tests**, not proven real tests avoided. **Mean simulated effort did not fall** versus the fixed strategy. The synthetic urgent-label classifier missed **74 of 328** urgent-labeled cases: its urgent-label precision was 78.2% and recall 77.4%, neither a validated emergency metric.

An example of actual held-out *simulation failure*: one pneumonia-labeled artificial case was predicted as viral respiratory illness after fatigue, headache, and a flagged artificial CRP result. We publish false positives, false negatives, a 21×21 confusion matrix, uncertainty intervals, counterfactual budget tests, and independent-seed stress tests in the generated artifact folder.

## Genuine public clinical record pilot — separate and exploratory

Sources: [MIMIC-IV Clinical Database Demo (2023), DOI 10.13026/dp1f-ex47](https://physionet.org/content/mimic-iv-demo/2.2/) and [MIMIC-IV-ED Demo (2023), DOI 10.13026/jzz5-vs76](https://physionet.org/content/mimic-iv-ed-demo/2.2/). Both are publicly available, de-identified **real** hospital record samples under [ODbL 1.0](https://physionet.org/content/mimic-iv-demo/view-license/2.2/).

The pilot examines **222 eligible admissions from 93 patients**. We split by patient rather than by admission: **60 patients / 135 admissions for training**, **14 / 48 for validation**, **19 / 39 for held-out test**. It models 12 **genuinely recorded** early laboratory abnormal-flag indicators and broad primary ICD discharge chapters. The public replay uses only **validation patients**; patients in the test cohort are not exposed through the demo.

| Held-out retrospective replay | Adaptive | Fixed order |
|---|---:|---:|
| Accuracy classifying broad discharge-diagnosis chapter | **23.1%** | **30.8%** |
| Mean historically recorded laboratory indicators revealed | 4.0 | 4.0 |

**Adaptive was worse on the real pilot.** This result is not hidden or substituted with synthetic accuracy. The study is tiny and selected; the model does not classify arbitrary real hospital patients reliably. It only replays tests **known in hindsight to have been performed**. No test is invented when absent, and missing is not treated as negative. Because historically unperformed test outcomes and clinical necessity are unknown, there is **selection bias** and no reliable counterfactual cost or test-reduction evaluation. Primary discharge groups are imperfect proxies for correct diagnosis.

The Health Track says **use public datasets only** and cautions against real patient records. These PhysioNet demos are openly released and de-identified, not private hospital records, but **organizer confirmation of acceptability has not been obtained**. This public pilot should be described as a supplementary research investigation, not a claim that NEXTQUESTION is safe for clinical deployment.

## Public symptom-only benchmark

The independent [Disease-Symptom Dataset](https://github.com/hemasriram111/Disease-Symptom-Dataset) contains 4,920 source rows but only **304 unique symptom patterns**. We deduplicate *before* splitting: 222 training, 41 validation, 41 held-out patterns. On this limited, templated dataset, adaptive symptom questioning reached 97.6% top-1 accuracy with 4.51 questions, versus 75.6% and 7.90 for fixed ordering. **Those results do not involve blood tests, imaging, actual medical charges, or verified clinical encounter outcomes.** The original source has no clearly declared data license; our code fetches it and does not redistribute its CSV.

## Why this matters and what is novel

The intended user is a **clinician deciding which information is missing** after a patient's initial presentation and existing investigations. Traditional disease classifiers output a probability from a fixed feature vector. NEXTQUESTION explores a different algorithmic step: **actively choose the next informative measurement under constraints**, update when it arrives, and distinguish simulated effort from medical costs. The project addresses concerns about test overuse and financial barriers to care; it **does not establish** fewer real hospital tests, earlier diagnoses, lower bills, or lives saved.

## Reproducibility, quality assurance, and attribution

- **Code:** hospital/engine.py — synthetic method; hospital/audit.py — confusion matrices, holdout errors and robustness; hospital/mimic_real.py — real public EHR patient-level experiment; hospital/real_ui.py — separate real-record explorer; hospital/public_benchmark.py — symptom-only benchmark.
- **Application:** app.py — responsive Gradio UI with hidden-result acquisition, manual prior findings, safeguards, and reset.
- **Testing:** tests/ — model/split/unit tests, hidden-result leakage checks, real-record replay tests, and Chromium browser checks for desktop/mobile.
- **Deployment:** Dockerfile and GitHub Actions CI include a real HTTP endpoint and recorded browser demonstrations. View latest verification in [GitHub Actions](https://github.com/ahblxz/NEXTQUESTION-Veritas-2026/actions).
- **Generated files:** artifacts/hospital/ contains the synthetic evaluation and error report; artifacts/mimic_real/ contains the separate real-record pilot model and evaluation; data source tables are excluded from release images.

**Dependencies and development:** Python, NumPy, Gradio, pytest, Playwright, Docker, GitHub Actions, Google Colab; dataset credits and links above. AI coding assistance included OpenAI ChatGPT and Anthropic Claude. No pretrained clinical model, paid clinical API or private patient database was used.

### Limitations

Synthetic likelihoods and effort points are not actual test performance or prices. In the real-record pilot, diagnoses are coarse discharge-code categories, histories are incomplete, and the available-test list has hindsight bias. The public sample is very small, adverse cases are poorly recognized, and no algorithm is validated to safely omit diagnostic workup. Any future medical use would require independent clinician review, institutional approvals, outcome-based testing and real financial measurements.

**This project is a prototype and is not medical advice.**
