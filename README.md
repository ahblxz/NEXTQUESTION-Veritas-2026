# NEXTQUESTION
### Which diagnostic information should a clinician investigate next?

**Veritas AI International High School Hackathon 2026 · AI + Health**

NEXTQUESTION is a **clinician-facing, multi-condition educational simulator**. Given a *fictional* presentation and already available *fictional* findings, it ranks possible conditions, estimates the expected diagnostic information provided by additional tests, then updates those rankings when a result is revealed. Our research question is whether adaptive information gathering can reduce the **number of artificial tests** needed to achieve similar diagnostic classification performance.

**This project is a prototype and is not medical advice.** It is **not a medical device**. Never enter real patient information or use its rankings to order, omit or postpone an actual medical test. Our diagnostic labels, findings, priors, and effort points are synthetic, not clinically validated.

## Run the actual demo

**[Open Google Colab — install, evaluate, launch](https://colab.research.google.com/github/ahblxz/NEXTQUESTION-Veritas-2026/blob/main/notebooks/02_run_complete_project.ipynb)**

Choose **Runtime → Run all**. When the final cell opens, click the temporary \`gradio.live\` URL. Keep Colab running. Or run the same application locally:

\`\`\`bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_hospital.py tests/test_audit.py
python hospital_train.py --out artifacts/hospital --n 4800
python -m hospital.audit
python app.py
\`\`\`

To run it as a deployable educational web service:

\`\`\`bash
docker build -t nextquestion .
docker run --rm -p 7860:7860 nextquestion
\`\`\`

Open \`http://localhost:7860\`. The Docker image is verified in CI; a **permanent public web host has not been deployed**.

## Main finding (locked synthetic holdout, 960 cases)

| Policy | Classification accuracy | Average simulated tests | Average hypothetical effort | Urgent-label false negatives | Urgent-label false positives |
|---|---:|---:|---:|---:|---:|
| **Adaptive NEXTQUESTION** | **73.6%** | **2.79** | **7.31 pts** | **74** | **71** |
| Fixed order | 70.0% | 3.53 | 7.31 pts | 94 | 69 |
| Cheapest first | 67.2% | 3.66 | 7.08 pts | 94 | 73 |
| Full artificial panel | 76.9% | 16.00 | 82.00 pts | 63 | 58 |

Adaptive selected approximately **21% fewer simulated test indicators** than fixed ordering and achieved +3.6 percentage points synthetic accuracy, but **did not reduce mean simulated effort**. The urgent/non-urgent confusion matrix for adaptive is **[[TN 561, FP 71], [FN 74, TP 254]]**. Urgent-label precision was **78.2%** and recall **77.4%**. These are **not** real-world emergency triage metrics.

**Honest failure:** In one held-out synthetic case, the true generated label was *Pneumonia* (urgent), but the model predicted *Viral respiratory illness* (non-urgent), after observing only the artificial symptoms fatigue/headache and a flagged CRP indicator. Possible contributors include overlapping symptom assumptions, noisy artificial findings and limited test acquisition. **It would be unsafe to treat the model's classification as clinical advice.**

## How it works — and our original addition

- **Who would use the concept:** A doctor reviewing a patient's presenting complaint and previous results before considering additional tests. The actual software is strictly a demonstration using invented scenarios.
- **Inputs:** Fictional symptoms, red flags and optionally an existing, invented binary test result. Free text is conservative keyword matching; the symptom checkboxes are more reliable.
- **Model:** Naïve Bayes with smoothed class-conditional symptom and test probabilities, learned from synthetic training records. Unknown symptoms are not automatically assumed absent.
- **Original contribution:** Adaptive test acquisition, **not a disease-specific classifier**. At each step the model estimates every unopened test's expected reduction in entropy and ranks by a validation-selected trade-off with illustrative test effort. Test results remain hidden until acquired; each one updates the differential and next recommendation.
- **Safety:** Fictional urgent warning flags override any cost-optimization story; the software never automatically cancels a test. Actual emergency assessment is outside its capabilities.
- **Evaluation protocol:** 4,800 generated cases; **3,120 training / 720 validation / 960 final held-out**. Stopping threshold and effort exponent are selected using validation only. **All demonstration cases are from validation**, never the final test set. Each run saves the per-class confusion matrix, 2×2 urgent-label confusion, 21 one-vs-rest error counts, failure examples and paired uncertainty intervals.

### Independent stress tests (no model retuning)

Using independently generated synthetic evaluation cohorts:
- **30% of reported positive symptoms suppressed:** adaptive accuracy **67.4%**, fixed **62.0%**.
- **10% of binary test findings flipped:** adaptive **67.5%**, fixed **64.9%**.
- **Different urgent-class prevalence:** adaptive **72.8%**, fixed **68.3%**.
- **Different hypothetical test costs:** adaptive **74.1%**, fixed **69.7%**.
- **New random seed under original assumptions:** adaptive **74.6%**, fixed **70.1%**.

These stress tests still use variations of the **same hand-designed artificial data generator**, so they are **not independent medical or hospital validation**.

### Supplemental public-data benchmark

The AI + Health brief says **“Use public datasets only.”** To add a publicly sourced experiment, \`hospital/public_benchmark.py\` downloads and benchmarks the public [Disease-Symptom Dataset](https://github.com/hemasriram111/Disease-Symptom-Dataset) at a pinned commit. **4,920 CSV rows shrink to 304 unique symptom patterns** after deduplication. We split by unique symptom evidence, with 222 training, 41 validation and 41 held-out patterns, preventing exact duplicates crossing splits.

On that **small, highly templated, public symptom-only benchmark**, adaptive symptom questioning achieved **97.6% accuracy with 4.51 questions** versus fixed-order questioning's **75.6% with 7.90 questions**. This is **not a diagnostic laboratory test benchmark**: the dataset has symptoms and disease labels, but **no lab-test findings, prices or care appropriateness labels**. Its original licensing is not clearly documented, so the raw CSV is fetched from the source and **not redistributed**. The main 21-condition lab-test simulation is still independently generated; judges should assess the track-specific public-data requirement with this distinction in mind.

### Motivating evidence (not evidence of our product's benefit)

- A [2021 systematic review in BMJ Quality & Safety](https://pubmed.ncbi.nlm.nih.gov/33972387/) analyzed 118 estimates of diagnostic test overuse; estimates varied widely across conditions and definitions.
- [Peterson–KFF's March 2026 analysis](https://www.healthsystemtracker.org/chart-collection/cost-affect-access-care/) reports that around **17% of US adults** delayed or went without some health care because of cost in 2024. This is a **US** statistic, not a UAE estimate.
- [Synthea](https://github.com/synthetichealth/synthea) documents synthetic multi-condition clinical data and informs future research directions. **We did not claim to train on Synthea records.**

These sources establish that unnecessary tests and affordability are real concerns; they do **not** establish that NEXTQUESTION safely avoids clinical tests, improves diagnosis timing, reduces actual bills or saves lives.

## Reproducibility, release and source credits

| Source file | Contents |
|---|---|
| \`hospital/engine.py\` | Simulated dataset, Bayesian differential, entropy-based test ranking, baseline evaluation |
| \`hospital/audit.py\` | Held-out FP/FN and confusion matrices, safety-label errors, actual simulation failure examples, robustness |
| \`hospital/public_benchmark.py\` | Pinned public dataset, deduplication-first symptom benchmark |
| \`hospital_train.py\` | Reproduce locked synthetic benchmark and validation-selected settings |
| \`app.py\` | Interactive patient intake, differential, prior fictional findings, test ranking and results |
| \`tests/\` | Unit, leakage, split, app and real mobile/desktop browser tests |
| \`notebooks/02_run_complete_project.ipynb\` | Colab evaluation and temporary live demo |
| \`Dockerfile\`, \`.github/workflows/ci.yml\` | Deployable container and end-to-end CI |

All research outputs regenerate in \`artifacts/hospital/\`: \`metrics.json\`, \`audit_metrics.json\`, \`audit_report.md\`, \`results.md\`, \`public/metrics.json\` and deterministic demo scenarios. The [GitHub Actions tab](https://github.com/ahblxz/NEXTQUESTION-Veritas-2026/actions) provides reproducible logs and a downloadable results/screenshot artifact.

**Libraries and tools:** Python 3.11, NumPy, Gradio, pytest, Playwright, Docker, GitHub Actions, Google Colab; public symptom benchmark source linked above; AI development assistance included [OpenAI ChatGPT](https://chatgpt.com/) and [Anthropic Claude](https://claude.ai/). No pretrained clinical model, paid clinical API or private patient database was used. All tools actually used by team members must be disclosed on the submission form.

## Important limitations

A correct result on **fabricated** diagnoses is not a clinically valid diagnosis. Test prices are made-up effort points. The full panel is an artificial comparison, not standard medical practice. The model covers only 21 artificial labels and can miss urgent-labeled cases. The text parser may mishandle unusual language or negation. This demo does not interpret actual numerical lab ranges or clinical guidelines and is not suitable for use with patients. Deployment as educational software is separate from medical-device approval.

**This project is a prototype and is not medical advice.**
