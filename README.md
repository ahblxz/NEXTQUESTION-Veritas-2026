# NEXTQUESTION

**AI + Health · Veritas AI International High School Hackathon 2026**

A runnable, multi-condition **synthetic research prototype** exploring cost-aware diagnostic information acquisition. It ranks possible explanations for a fictional presentation, scores which still-unobserved test is expected to reduce diagnostic uncertainty most, and updates after new synthetic test results. The clinician-facing interface offers preset and fictional manual scenarios.

**Not a medical device. Not medical advice. Do not enter real patient information or use this prototype to order, omit or delay clinical tests.**

## Demo (Google Colab)

[Open the runnable notebook](https://colab.research.google.com/github/ahblxz/NEXTQUESTION-Veritas-2026/blob/main/notebooks/02_run_complete_project.ipynb) → **Runtime → Run all** → open the temporary `gradio.live` URL in the final cell. Keep the Colab runtime running while using the demo.

## Run and test locally

Requires Python 3.11+.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_hospital.py
python hospital_train.py --out artifacts/hospital --n 4800
python app.py
```

## Deploy as a web demo

Build and run the included Docker container:

```bash
docker build -t nextquestion .
docker run --rm -p 7860:7860 nextquestion
```

Open `http://localhost:7860`. Deploy the same Docker image to a container host exposing port 7860. Hosting depends on the provider; the repository itself is **not** a continuously hosted demo.

## Source and verification

| File | Purpose |
|---|---|
| `hospital/engine.py` | Multi-condition synthetic generator, Bayesian classifier, adaptive information-gain testing and baseline comparison |
| `hospital_train.py` | Reproducible 4,800-case train/validation/held-out experiment |
| `app.py` | Interactive Gradio dashboard |
| `notebooks/02_run_complete_project.ipynb` | One-click Colab setup, tests, evaluation and demo |
| `tests/test_hospital.py` | Model, data-leakage, cost-boundary and safety checks |
| `tests/browser_smoke.py` | HTTP launch and phone-sized browser integration test |
| `Dockerfile` | Self-contained web-demo build |
| `.github/workflows/ci.yml` | Automatic end-to-end verification |

The code generates all model and experiment artifacts locally; no prebuilt model or private records are required.

**Data:** 4,800 artificial scenarios, 21 synthetic syndrome labels in seven categories and 16 binary artificial test findings. Associations and hypothetical effort weights are hand-designed illustrative assumptions, **not** clinically measured prevalence, sensitivity, specificity, prices or treatment recommendations. Outcomes cannot demonstrate real-world diagnostic safety or financial savings.

**Implementation and credits:** Python, NumPy, Gradio, pytest, Playwright, Docker, GitHub Actions. Code development included AI coding assistance (OpenAI ChatGPT; disclose any other tools used in the submission form). No third-party patient dataset or pretrained diagnosis model is used.
