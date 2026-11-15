"""NEXTQUESTION · Multi-condition diagnostic test prioritization research dashboard.

Educational synthetic simulation ONLY. Not a medical device or medical advice.
NEVER enter real patient-identifiable information.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import json
import os

import gradio as gr
import numpy as np

from hospital.engine import (
    CONDITIONS, COSTS, RED_FLAGS, SEED, SYMPTOMS, SYNTHETIC_NOTE, TESTS,
    URGENCY, BayesTestModel, extract_symptoms, load_model, save_research,
)
from hospital.real_ui import build_pilot, READY as REAL_PILOT_READY

ART = Path("artifacts/hospital")
if not (ART / "model.json").exists() or not (ART / "demo_cases.json").exists():
    save_research()
MODEL, THRESHOLD = load_model()
SCENARIOS = json.loads((ART / "demo_cases.json").read_text(encoding="utf-8"))
BY_SCENARIO = {x["id"]: x for x in SCENARIOS}
SIMULATED_CASE = "Synthetic demonstration"
MANUAL_CASE = "Manual fictional case"
SYNTHETIC_WARNING = "Research simulator only · Do not enter real patient data · Not medical advice"
FULL_COST = sum(COSTS.values())
DEPT_COUNT = len(set(c[1] for c in CONDITIONS))
EVALUATION = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
ADAPTIVE_SCORE = EVALUATION["test_results"]["adaptive"]
FIXED_SCORE = EVALUATION["test_results"]["fixed"]

CSS = """
/* NEXTQUESTION — one visual system for both Gradio light and dark preferences */
:root { color-scheme: dark; }
html, body, gradio-app, #root, .gradio-container {
  background: #070d17 !important; color: #e9f4fb !important;
}
body { margin:0; }
/* Keep the competition UI focused; framework credits remain in README. */
.gradio-container footer {display:none !important;}
.gradio-container {
  max-width: 1420px !important; padding: 12px 12px 24px !important;
  margin: 0 auto !important;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  --body-background-fill: #070d17 !important;
  --block-background-fill: #101e2d !important;
  --input-background-fill: #14283b !important;
  --body-text-color: #e9f4fb !important;
  --block-label-text-color: #b8cfdd !important;
  --block-border-color: #253c50 !important;
  --input-border-color: #37566a !important;
}
.gradio-container, .gradio-container * { box-sizing: border-box; }
/* Default Gradio shells consume 64px of extra width. Override them at the
   root level as well, so browsers that prioritize the base theme still shrink. */
.gradio-container .main.fillable {
  padding: 8px 8px !important;
  max-width:100% !important;min-width:0 !important;
}
.gradio-container .main.fillable > .wrap,
.gradio-container .main.fillable main.contain {
  padding:0 !important;margin:0 !important;
  max-width:100% !important;min-width:0 !important;
}
.gradio-container .prose,
.gradio-container .prose h1, .gradio-container .prose h2,
.gradio-container .prose h3, .gradio-container .prose p {
  color: #edf8ff !important;
}
.gradio-container .prose h3 {
  font-size: 13px !important; font-weight: 750 !important;
  letter-spacing: .12em !important; text-transform: uppercase;
  margin: 3px 0 12px !important; color:#93afc3 !important;
}
/* Branded editorial header */
.nq-hero {
  position:relative; isolation:isolate; overflow:hidden;
  background:linear-gradient(112deg,#101e32 2%,#102d39 68%,#092a36 100%);
  border:1px solid #2b4b5f; border-radius:20px;
  padding:28px 34px 27px; margin:12px 0 14px;
  box-shadow:0 14px 45px rgba(0,0,0,.18);
}
.nq-hero::before {
  content:""; pointer-events:none; position:absolute; z-index:-1;
  right:-100px; top:-200px; width:410px; height:410px;
  background:radial-gradient(circle,rgba(44,218,199,.19),transparent 66%);
}
.nq-hero::after {
  content:"";pointer-events:none;position:absolute;z-index:-1;
  width:340px;height:340px;right:40px;bottom:-300px;
  background:radial-gradient(circle,rgba(56,123,215,.15),transparent 69%);
}
.nq-brand {display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 22px;}
.nq-brand-left {display:flex;align-items:center;gap:11px;}
.nq-mark {
  display:grid;place-items:center; width:33px;height:33px;border-radius:10px;
  background:#1ad0b8;color:#052324 !important; font-size:23px;font-weight:900;
  box-shadow:0 0 0 5px rgba(30,208,183,.12);
}
.nq-wordmark {font-size:15px;letter-spacing:.10em;font-weight:850;color:#f6fffe !important;}
.nq-tagline {
  display:inline-flex;align-items:center;gap:7px;border:1px solid #42736f;
  color:#91e5d8 !important;border-radius:999px;padding:6px 12px;
  letter-spacing:.10em;font-size:10px;font-weight:800;white-space:nowrap;
}
.nq-tagline::before {content:"";width:6px;height:6px;background:#44f2cf;border-radius:50%;}
.nq-hero h1 {
  margin:0 0 10px; max-width:790px;color:#f7feff !important;
  font-size:clamp(28px,3.8vw,46px);font-weight:850;line-height:1.10;letter-spacing:-.043em;
}
.nq-hero p { color:#abc5d2 !important; max-width:800px; line-height:1.65;
  font-size:14px;margin:0; }
.nq-accent {color:#58e6d0 !important;}
.nq-hero-stats {
  display:flex;gap:10px;flex-wrap:wrap;margin:19px 0 0;
}
.nq-hero-stat {
  border:1px solid rgba(136,200,203,.22);background:rgba(9,27,42,.50);
  padding:7px 12px;border-radius:8px;color:#bed4df !important;font-size:11px;
}
.nq-hero-stat b {font-size:12px;color:#e6fcf7 !important;}
/* A single calm benchmark bar; not a second giant hero */
.nq-benchmark {
  display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px;margin:0 0 14px;
}
.nq-benchmark .nq-bench-tile {
  background:#101e2a;border:1px solid #273d4c;border-radius:12px;
  padding:13px 16px;
}
.nq-bench-label {display:block;color:#91adbd !important;font-size:11px;line-height:1.3;}
.nq-bench-value {display:block;color:#f1fafa !important;font-size:22px;
  font-weight:850;letter-spacing:-.03em;margin-top:3px;}
.nq-benchmark small {font-size:10px;color:#8cabbc !important;}
.nq-benchmark-foot {grid-column:1/-1;margin:-1px 0 2px;color:#90aebd !important;font-size:11px;}
.nq-safety {
  padding:11px 14px;margin:0 0 18px;background:#282118;
  color:#ead4ac !important;border:1px solid #5b4936;border-radius:10px;
  font-size:12px;line-height:1.55;
}
.nq-safety strong {color:#ffe3b3 !important;}
/* Main columns are full dark panels; no floating white field cards */
.nq-intake-column {
  background:#0d1928 !important;border:1px solid #273e50;border-radius:18px;
  padding:18px 20px 24px;box-shadow:0 18px 42px rgba(0,0,0,.16);
  align-self:flex-start;
}
.nq-insights-column {padding:0 0 0 8px;}
.nq-intake-column .block,
.nq-intake-column .form,
.nq-intake-column .gr-block,
.nq-intake-column .gr-form,
.nq-intake-column .container {
  background:#132536 !important;color:#eaf6ff !important;
  border-color:#2b465a !important;
}
.nq-intake-column .gradio-row, .nq-intake-column .gradio-column,
.nq-intake-column .nq-no-card {
  background:transparent !important;
}
.nq-intake-column .wrap,
.nq-intake-column .input-container,
.nq-intake-column [role="combobox"] {
  background:#172c3e !important; color:#e8f4ff !important;
}
.nq-intake-column label,
.nq-intake-column .block-label,
.nq-intake-column .label-wrap {
  color:#b8cedb !important;
}
.gradio-container input:not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea,.gradio-container select,
.gradio-container [role="combobox"] {
  background:#162b3d !important;color:#f3fbff !important;
  border:1px solid #34536a !important;border-radius:9px !important;
  box-shadow:none !important;outline-color:#40e3cb !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {color:#7f9bad !important;opacity:1;}
.gradio-container input:focus, .gradio-container textarea:focus,
.gradio-container [role="combobox"]:focus-within {
  border-color:#26dbbf !important;box-shadow:0 0 0 2px rgba(29,214,186,.12) !important;
}
#nq-mode label:has(input[type="radio"]),
#nq-finding label:has(input[type="radio"]) {
  background:#15283a !important;color:#d5e7f2 !important;
  border:1px solid #314a5e !important;border-radius:9px !important;
  padding:10px 12px !important;
}
#nq-mode label:has(input[type="radio"]:checked),
#nq-finding label:has(input[type="radio"]:checked) {
  background:#113c40 !important;border-color:#25cdb9 !important;color:#f1fffc !important;
}
.gradio-container input[type="radio"], .gradio-container input[type="checkbox"] {
  accent-color:#21d2bc !important;
}
#nq-symptoms .wrap, #nq-flags .wrap, #nq-manual-test .wrap {
  min-height:43px;background:#14283b !important;
  border:1px solid #34536a !important;border-radius:10px !important;
}
#nq-symptoms .token, #nq-flags .token, #nq-symptoms [class*="token"],
#nq-flags [class*="token"] {
  background:#163d44 !important;color:#d7fff7 !important;border:1px solid #285e63 !important;
  border-radius:7px !important;
}
.gradio-container [role="option"], .gradio-container [role="listbox"],
.gradio-container .options {
  background:#142538 !important;color:#edf9ff !important;
}
.gradio-container [role="option"]:hover {background:#204656 !important;}
#nq-existing label:has(input[type="checkbox"]) {
  background:transparent !important;color:#d5e7ef !important;
}
.nq-intake-hint {color:#82a1b3 !important;font-size:12px;margin:0 0 12px;line-height:1.55;}
.nq-step-title {
  display:flex;gap:9px;align-items:center;font-size:11px;
  color:#65e4ce !important;font-weight:850;letter-spacing:.10em;text-transform:uppercase;
  margin:13px 0 9px;
}
.nq-step-title span {width:22px;height:22px;display:grid;place-items:center;
  background:#184248;color:#82fff0 !important;border-radius:6px;font-size:11px;}
.nq-section-divider {height:1px;background:#293d4c;margin:16px 0;}
.nq-auto-controls,.nq-manual-controls {
  margin-top:12px;background:transparent !important;border:0 !important;
  box-shadow:none !important;padding:0 !important;
}
.nq-auto-controls > div,.nq-manual-controls > div {
  background:transparent !important;border-color:transparent !important;
}
/* Buttons – consistent height, strong contrast, no default pale white states */
.gradio-container button {
  min-height:44px !important;border-radius:10px !important;
  font-weight:750 !important;letter-spacing:.005em;
  transition:transform .16s ease,box-shadow .16s ease,background .16s ease !important;
}
.gradio-container button.primary {
  color:#042b2e !important;background:#29dac5 !important;
  border:1px solid #46ead5 !important;
  box-shadow:0 5px 18px rgba(31,209,186,.16) !important;
}
.gradio-container button.primary:hover:not(:disabled) {
  background:#57f2df !important;transform:translateY(-1px) !important;
}
.gradio-container button.secondary {
  color:#d8eaf5 !important;background:#1b3246 !important;
  border:1px solid #37556a !important;
}
.gradio-container button.secondary:hover:not(:disabled) {
  background:#25455d !important;
}
.gradio-container button:disabled, .gradio-container button[disabled] {
  background:#182b3a !important;color:#8ca5b6 !important;
  border-color:#263e50 !important;opacity:1 !important;box-shadow:none !important;
  cursor:not-allowed !important;
}
/* Results are one consistent family of glass-dark data surfaces */
.nq-card {
  background:linear-gradient(130deg,#122538,#102030) !important;
  border:1px solid #2a4355 !important;border-radius:15px;
  padding:18px 20px;margin:0 0 13px;box-shadow:0 12px 28px rgba(0,0,0,.10);
}
.nq-card *, .nq-card p, .nq-card div, .nq-card span {
  color:#dcecf5;opacity:1 !important;
}
.nq-card h3 {
  margin:0 0 13px !important;color:#f2fcff !important;
  font-size:16px;font-weight:800;letter-spacing:-.02em;
}
.nq-card .nq-muted {font-size:12px;color:#97b5c7 !important;line-height:1.55;}
.nq-card .nq-emphasis {color:#5debd1 !important;font-weight:850;}
.nq-card .nq-warning {color:#ffe4ad !important;font-weight:750;}
.nq-line {
  display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:12px;padding:11px 0;
  border-bottom:1px solid #2a4354;font-size:12px;line-height:1.45;
}
.nq-line:last-child {border-bottom:0;}
.nq-pill {
  background:#1c3d4a !important;border:1px solid #34606a;
  padding:5px 9px;border-radius:999px;color:#c1f7ea !important;
  font-weight:700;font-size:11px;white-space:nowrap;
}
.nq-bar {
  height:5px;background:#2b4558;border-radius:100px;overflow:hidden;margin:8px 0 0;
}
.nq-bar > span {display:block;height:100%;background:linear-gradient(90deg,#16cdbb,#7deade);}
.nq-step {border-left:2px solid #2ccfba;padding:10px 0 10px 17px;margin:9px 0;}
.nq-step strong {color:#6df1d9 !important;}
.nq-badge {
  background:#312a1c;border:1px solid #80653e;color:#ffe4b5 !important;
  border-radius:10px;padding:10px 12px;margin:10px 0;font-size:12px;line-height:1.5;
}
.nq-metric {font-size:34px;font-weight:900;letter-spacing:-.04em;color:#f4fffd !important;}
.nq-foot {
  margin:24px 0 10px;padding:14px 0 0;border-top:1px solid #243847;
  font-size:11px;color:#8ea9bb !important;line-height:1.65;
}
/* Small screens: no horizontal squeeze or overpacked tri-column fields */
@media(max-width:1000px) {
  .gradio-container {padding:12px 15px 28px !important;}
  .nq-insights-column {padding:0 !important;}
  .nq-intake-column {padding:15px !important;}
}
@media(max-width:670px) {
  /* Gradio's default wrapper can inherit a 500px min-content width. */
  .gradio-container {
    width:100vw !important;max-width:100vw !important;min-width:0 !important;
    padding:8px 10px 20px !important;overflow-x:clip !important;
  }
  .gradio-container .main.fillable,.gradio-container .main,
  .gradio-container main.contain,.gradio-container .wrap,
  .gradio-container .column,.gradio-container .html-container,
  .gradio-container .prose {
    width:100% !important;max-width:100% !important;min-width:0 !important;
  }
  /* Gradio wraps the page in two extra padded shells. Eliminate those
     on mobile so the medical dashboard actually uses the phone screen. */
  .gradio-container .main.fillable,
  .gradio-container .main.fillable > .wrap,
  .gradio-container main.contain {
    padding:0 !important;margin:0 !important;min-width:0 !important;
  }
  .gradio-container .block:has(.html-container) {
    padding:0 !important;background:transparent !important;border:0 !important;
  }
  .gradio-container .html-container,.gradio-container .prose {
    padding:0 !important;margin:0 !important;
  }
  .nq-brand {flex-wrap:wrap !important;justify-content:flex-start !important;}
  .nq-tagline {white-space:normal !important;max-width:100% !important;}
  .nq-hero h1 {overflow-wrap:anywhere !important;}
  .nq-hero {padding:18px !important;border-radius:15px;}
  .nq-hero h1 {font-size:29px !important;}
  .nq-hero p {font-size:12px;}
  .nq-brand {margin-bottom:20px;}
  .nq-tagline {font-size:9px;padding:5px 8px;}
  .nq-hero-stats {gap:6px;}
  .nq-hero-stat {font-size:10px;padding:6px 9px;}
  .nq-benchmark {grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;}
  .nq-benchmark .nq-bench-tile {padding:10px 12px;}
  .nq-bench-value {font-size:19px;}
  .nq-intake-column {padding:13px !important;margin:0 0 12px;}
  .nq-insights-column {padding:0 !important;}
  .nq-card {padding:13px !important;border-radius:12px;}
  .nq-line {align-items:flex-start;}
  /* Force a true single-column layout on phones; Gradio's nested flex items
     otherwise retain an intrinsic width and cause horizontal scrolling. */
  .nq-main-layout {display:flex !important;flex-direction:column !important;
    align-items:stretch !important;flex-wrap:wrap !important;
    min-width:0 !important;max-width:100% !important;width:100% !important;}
  .nq-main-layout > *,.nq-intake-column,.nq-insights-column {
    min-width:0 !important;max-width:100% !important;
    width:100% !important;flex:0 1 auto !important;}
  .nq-intake-column > *,.nq-insights-column > * {
    min-width:0 !important;max-width:100% !important;}
  #nq-symptoms,#nq-flags,#nq-manual-test {min-width:0 !important;max-width:100% !important;}
  #nq-symptoms .wrap,#nq-flags .wrap {min-width:0 !important;
    max-width:100% !important;flex-wrap:wrap !important;overflow-x:clip !important;}
  .nq-benchmark .nq-bench-tile {min-width:0 !important;}

  /* Gradio 6's wrapper rules are unusually specific; these explicit overrides
     prevent the built-in 30px + 32px padding shells consuming the phone screen. */
  .gradio-container.gradio-container.gradio-container {
    padding:8px 10px 20px !important;
    width:100vw !important;max-width:100vw !important;
  }
  .gradio-container.gradio-container .main.fillable.fillable {
    padding:0 !important;margin:0 !important;
    width:100% !important;max-width:100% !important;
  }
  .gradio-container.gradio-container .main.fillable.fillable > .wrap {
    padding:0 !important;margin:0 !important;
    width:100% !important;max-width:100% !important;
  }
  .gradio-container.gradio-container .main.fillable.fillable main.contain {
    padding:0 !important;margin:0 !important;
    width:100% !important;max-width:100% !important;
  }
}
"""
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_sm,
).set(
    body_background_fill="#070d17",
    body_background_fill_dark="#070d17",
    block_background_fill="#101e2d",
    block_background_fill_dark="#101e2d",
    input_background_fill="#14283b",
    input_background_fill_dark="#14283b",
    body_text_color="#e9f4fb",
    body_text_color_dark="#e9f4fb",
    block_label_text_color="#b8cfdd",
    block_label_text_color_dark="#b8cfdd",
    block_border_color="#2d4559",
    block_border_color_dark="#2d4559",
    input_border_color="#365568",
    input_border_color_dark="#365568",
    button_primary_background_fill="#29dac5",
    button_primary_background_fill_dark="#29dac5",
    button_primary_text_color="#092e2f",
    button_primary_text_color_dark="#092e2f",
    button_secondary_background_fill="#1b3246",
    button_secondary_background_fill_dark="#1b3246",
    button_secondary_text_color="#e8f6ff",
    button_secondary_text_color_dark="#e8f6ff",
    checkbox_background_color="#173345",
    checkbox_background_color_dark="#173345",
)

def _blank(title, message):
    return (f'<div class="nq-card"><h3>{escape(title)}</h3>'
            f'<p class="nq-muted">{escape(message)}</p></div>')


def case_input(scenario):
    case = BY_SCENARIO.get(scenario)
    if not case:
        return "", []
    return case["complaint"], case["symptoms"]


def _active(input_symptoms, complaint):
    keys = set(input_symptoms or []) | set(extract_symptoms(complaint or ""))
    return sorted(key for key in keys if key in SYMPTOMS)


def _render(state):
    if not state:
        return (
            _blank("Assessment status", "Choose an example or enter a fictional complaint, then start."),
            _blank("Differential considerations", "Candidate conditions appear here. They are not diagnoses."),
            _blank("Suggested next tests", "Research-only test ranking appears after intake."),
            _blank("Test pathway", "Results are added one test at a time."),
            _blank("Illustrative test burden", "Synthetic effort points; not medical prices."),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    symptoms, known = state["symptoms"], state["obtained"]
    posterior = MODEL.posterior(symptoms, known)
    candidates = MODEL.candidates(symptoms, known, top=5)
    progress = len(known)
    research_stop = progress > 0 and float(posterior.max()) >= THRESHOLD
    flags = state.get("flags", [])
    urgent_mass = float(posterior[URGENCY].sum())
    alert = bool(flags or {"chest_pain", "breathlessness", "weakness"} & set(symptoms))
    banner = (
        '<div class="nq-badge"><strong>Priority clinical review:</strong> The reported '
        'features may need urgent assessment. Do not use this simulated ranking to '
        'delay evaluation, skip testing, or exclude emergencies.</div>'
        if alert else '<p class="nq-muted">This simulator cannot exclude an emergency '
        'or determine which tests can be safely omitted.</p>'
    )
    status = (
        '<div class="nq-card"><h3>Assessment overview</h3>'
        f'<div class="nq-line"><span>Reported symptom signals</span>'
        f'<span class="nq-pill">{len(symptoms)} recognized</span></div>'
        f'<div class="nq-line"><span>Investigations recorded</span><span class="nq-pill">{progress}</span></div>'
        f'<div class="nq-line"><span>Highest model weight</span>'
        f'<span class="nq-emphasis">{100 * float(posterior.max()):.1f}%</span></div>'
        f'{banner}<p class="nq-muted">Probabilities come from artificial training assumptions, '
        'not clinically calibrated risk estimates. Urgent-class model weight: '
        f'{urgent_mass:.0%} (not a triage assessment).</p></div>'
    )
    top = []
    for row in candidates:
        value = 100 * row["probability"]
        top.append(
            f'<div class="nq-line"><div style="flex:1;min-width:180px">'
            f'<strong>{escape(row["diagnosis"])}</strong>'
            f'<div class="nq-muted">{escape(row["category"])}'
            f'{" · Priority-review category" if row["urgent"] else ""}</div>'
            f'<div class="nq-bar"><span style="width:{value:.1f}%"></span></div>'
            f'</div><span class="nq-pill">{value:.1f}%</span></div>'
        )
    differential = ('<div class="nq-card"><h3>Possible explanations to investigate</h3>'
                    + "".join(top) + '<p class="nq-muted">Model-ranked simulation categories. '
                    'Not an exhaustive differential or an individual diagnosis.</p></div>')
    remaining = max(0, state["budget"] - state["cost"])
    ranking = MODEL.ranking(symptoms, known, budget=remaining)
    options = []
    for idx, row in enumerate(ranking[:4], 1):
        options.append(
            f'<div class="nq-line"><div><strong>{idx}. {escape(row["name"])}</strong>'
            f'<div class="nq-muted">Estimated entropy reduction: '
            f'{row["information"]:.3f} nats · Artificial positive finding '
            f'likelihood: {row["abnormal_likelihood"]:.0%}</div></div>'
            f'<span class="nq-pill">{row["cost"]} effort pts</span></div>'
        )
    if research_stop:
        explanation = (
            f"The research policy's validation-selected stopping threshold "
            f"({THRESHOLD:.0%} relative top-class weight) has been reached. "
            "Suggested acquisition pauses here; this NEVER means medically safe to stop testing."
        )
    elif ranking:
        explanation = ('Highest estimated reduction in diagnostic uncertainty per '
                       'synthetic effort point under the current budget.')
    else:
        explanation = 'No tests remain within the illustrative budget.'
    tests = ('<div class="nq-card"><h3>Ranked information-gathering options</h3>'
             + "".join(options)
             + f'<p class="nq-muted">{escape(explanation)} '
               'These are not test orders. A clinician must make all actual decisions.</p></div>')
    steps = ['<div class="nq-step"><strong>Initial presentation</strong>'
             f'<p class="nq-muted">{len(symptoms)} symptoms reported; no test results disclosed.</p></div>']
    for step in state["history"]:
        finding = "abnormal / flagged" if step["value"] else "not flagged"
        steps.append(
            f'<div class="nq-step"><strong>{escape(TESTS[step["key"]][0])}</strong>'
            f'<p>{escape(finding)} · +{step.get("incremental_cost", COSTS[step["key"]])} simulated pts '
            f'{" · already available" if step.get("previously_available") else ""}'
            f' · leading model weight {step["confidence"]:.1%}</p></div>'
        )
    timeline = ('<div class="nq-card"><h3>Sequential investigation history</h3>'
                + "".join(steps) + '</div>')
    burden = (
        '<div class="nq-card"><h3>Simulated investigation burden</h3>'
        f'<div class="nq-metric">{state["cost"]} <span style="font-size:14px">points used</span></div>'
        f'<div class="nq-line"><span>Tests recorded</span><strong>{progress} of {len(TESTS)} test types</strong></div>'
        f'<div class="nq-line"><span>Full hypothetical panel</span><strong>{FULL_COST} points</strong></div>'
        f'<div class="nq-line"><span>Remaining research budget</span><strong>{remaining} points</strong></div>'
        '<p class="nq-muted">The full panel is an artificial comparator, not standard hospital '
        'practice. The difference is NOT a medical saving or evidence any test is unnecessary.</p>'
        '</div>'
    )
    enabled = bool(ranking) and len(known) < 6 and not research_stop
    return status, differential, tests, timeline, burden, (
        gr.update(interactive=enabled and state["mode"] == SIMULATED_CASE)
    ), (
        gr.update(interactive=len(known) < 6 and state["mode"] == MANUAL_CASE)
    )


def start(mode, scenario, complaint, selected, flags):
    present = _active(selected, complaint)
    if not present:
        msg = ("No supported symptoms recognized. Choose symptom checkboxes or "
               "try a different fictional complaint. No test is recommended.")
        status, differential, tests, pathway, burden, one, two = _render(None)
        return None, _blank("Insufficient input", msg), differential, tests, pathway, burden, one, two
    if mode == SIMULATED_CASE and scenario not in BY_SCENARIO:
        return start(MANUAL_CASE, None, complaint, selected, flags)
    state = {"mode": mode, "scenario": scenario if mode == SIMULATED_CASE else None,
             "symptoms": present, "flags": list(flags or []),
             "obtained": {}, "history": [], "cost": 0, "budget": 22}
    return state, *_render(state)


def _add_result(state, finding, test_key=None, already_available=False):
    if not state or len(state["obtained"]) >= 6:
        return (state, *_render(state))
    if test_key is None:
        ranking = MODEL.ranking(state["symptoms"], state["obtained"],
                               state["budget"] - state["cost"])
        if not ranking:
            return (state, *_render(state))
        key = ranking[0]["key"]
    else:
        # Fictional cases can include pre-existing clinician-chosen test results.
        key = test_key
        if key not in TESTS or key in state["obtained"]:
            return (state, *_render(state))
    if finding not in (0, 1):
        raise ValueError("A synthetic test result must be abnormal or not flagged")
    additional_effort = 0 if already_available else COSTS[key]
    if state["cost"] + additional_effort > state["budget"]:
        return state, *_render(state)
    state = {**state, "obtained": dict(state["obtained"]),
             "history": list(state["history"])}
    state["obtained"][key] = finding
    state["cost"] += additional_effort
    confidence = float(MODEL.posterior(state["symptoms"], state["obtained"]).max())
    state["history"].append({
        "key": key, "value": finding, "confidence": confidence,
        "incremental_cost": additional_effort, "previously_available": bool(already_available),
    })
    return state, *_render(state)


def acquire(state):
    if not state or state.get("mode") != SIMULATED_CASE:
        return state, *_render(state)
    # Match the evaluation stopping rule; bypassing a disabled UI button
    # cannot silently acquire additional synthetic results.
    if state["obtained"] and float(MODEL.posterior(state["symptoms"], state["obtained"]).max()) >= THRESHOLD:
        return state, *_render(state)
    case = BY_SCENARIO[state["scenario"]]
    ranked = MODEL.ranking(state["symptoms"], state["obtained"],
                           state["budget"] - state["cost"])
    if not ranked:
        return state, *_render(state)
    key = ranked[0]["key"]
    return _add_result(state, int(case["test_findings"][key]))


def record_result(state, result, test_key=None, existing=False):
    if not state or state.get("mode") != MANUAL_CASE:
        return state, *_render(state)
    return _add_result(
        state, 1 if result == "Abnormal / flagged" else 0, test_key,
        already_available=bool(existing),
    )


def reset():
    return None, *_render(None)


with gr.Blocks(
    title="NEXTQUESTION | Diagnostic information research",
) as demo:
    gr.HTML(f"""
    <header class="nq-hero">
      <div class="nq-brand">
        <div class="nq-brand-left">
          <div class="nq-mark">+</div>
          <div class="nq-wordmark">NEXTQUESTION</div>
        </div>
        <span class="nq-tagline">RESEARCH DEMONSTRATION</span>
      </div>
      <h1>Find the <span class="nq-accent">next useful test.</span><br>Not every possible test.</h1>
      <p>A cost-aware diagnostic information simulator. Enter a fictional patient
      presentation; watch an adaptive model prioritize the next informative
      result and update its reasoning as evidence arrives.</p>
      <div class="nq-hero-stats">
        <span class="nq-hero-stat"><b>{DEPT_COUNT}</b> simulated clinical categories</span>
        <span class="nq-hero-stat"><b>{len(CONDITIONS)}</b> artificial condition labels</span>
        <span class="nq-hero-stat"><b>{len(TESTS)}</b> hypothetical tests</span>
        <span class="nq-hero-stat">Synthetic method demo + separate public-EHR pilot</span>
      </div>
    </header>
    """)
    gr.HTML(f"""
    <div class="nq-benchmark">
      <div class="nq-bench-tile">
        <span class="nq-bench-label">Adaptive accuracy</span>
        <span class="nq-bench-value">{ADAPTIVE_SCORE["accuracy"]:.1%}</span>
        <small>held-out synthetic cases</small>
      </div>
      <div class="nq-bench-tile">
        <span class="nq-bench-label">Fixed-order accuracy</span>
        <span class="nq-bench-value">{FIXED_SCORE["accuracy"]:.1%}</span>
        <small>same evaluation cohort</small>
      </div>
      <div class="nq-bench-tile">
        <span class="nq-bench-label">Adaptive tests per case</span>
        <span class="nq-bench-value">{ADAPTIVE_SCORE["mean_tests"]:.2f}</span>
        <small>simulated test indicators</small>
      </div>
      <div class="nq-bench-tile">
        <span class="nq-bench-label">Fixed-order tests per case</span>
        <span class="nq-bench-value">{FIXED_SCORE["mean_tests"]:.2f}</span>
        <small>simulated test indicators</small>
      </div>
      <p class="nq-benchmark-foot">The simulated effort cost was essentially equal
      ({ADAPTIVE_SCORE["mean_effort_points"]:.2f} vs
      {FIXED_SCORE["mean_effort_points"]:.2f} points). These results do not
      establish actual hospital savings, medical accuracy, or patient safety.</p>
    </div>
    """)
    gr.HTML(
        '<div class="nq-safety"><strong>Research only.</strong> This is not '
        'a clinical device or medical advice. All cases and test findings in '
        'this interface are fictional. Never use it to order, skip, or delay '
        'real medical testing or enter identifiable patient information.</div>'
    )

    gr.HTML(
        '<div class="nq-step-title"><span>S</span> Main method demonstration '
        '· generated fictional cases</div>'
    )
    with gr.Accordion(
        label="Explore real public de-identified hospital records (separate research pilot)",
        open=False,
    ):
        build_pilot()
    with gr.Row(equal_height=False, elem_classes=["nq-main-layout"]):
        with gr.Column(
            scale=5, min_width=325, elem_classes=["nq-intake-column"]
        ):
            gr.HTML(
                '<div class="nq-step-title"><span>01</span> Patient intake</div>'
                '<p class="nq-intake-hint">Choose a prepared scenario or enter '
                'a fictional complaint. Unselected symptoms remain unknown.</p>'
            )
            mode = gr.Radio(
                choices=[SIMULATED_CASE, MANUAL_CASE],
                value=SIMULATED_CASE,
                label="Assessment mode",
                elem_id="nq-mode",
            )
            scenario = gr.Dropdown(
                choices=list(BY_SCENARIO),
                value=SCENARIOS[0]["id"],
                label="Demonstration patient",
                elem_id="nq-scenario",
            )
            complaint = gr.Textbox(
                label="Presenting complaint",
                value=SCENARIOS[0]["complaint"],
                placeholder="Example: fever, cough, trouble breathing",
                lines=2,
                elem_id="nq-complaint",
            )
            selected = gr.Dropdown(
                choices=[(label, key) for key, label in SYMPTOMS.items()],
                value=SCENARIOS[0]["symptoms"],
                multiselect=True,
                filterable=True,
                label="Reported symptoms",
                info="Search or select multiple symptoms. Not selected = unknown.",
                elem_id="nq-symptoms",
            )
            flags = gr.Dropdown(
                choices=[(label, key) for key, label in RED_FLAGS.items()],
                value=[],
                multiselect=True,
                filterable=True,
                label="Urgent warning features",
                info="Select only if included in the fictional presentation.",
                elem_id="nq-flags",
            )

            gr.HTML('<div class="nq-section-divider"></div>'
                    '<div class="nq-step-title"><span>02</span> Begin assessment</div>')
            with gr.Row():
                btn_start = gr.Button(
                    "Start assessment", variant="primary", elem_id="nq-start",
                )
                btn_reset = gr.Button(
                    "Reset", variant="secondary", elem_id="nq-reset",
                )

            with gr.Group(
                elem_classes=["nq-auto-controls"], visible=True
            ) as simulated_panel:
                gr.HTML(
                    '<p class="nq-intake-hint">After assessment, reveal the '
                    'AI-ranked test result to update the model.</p>'
                )
                btn_next = gr.Button(
                    "Simulate suggested test result",
                    variant="secondary",
                    interactive=False,
                    elem_id="nq-acquire",
                )

            with gr.Group(
                elem_classes=["nq-manual-controls"], visible=False
            ) as manual_panel:
                gr.HTML(
                    '<p class="nq-intake-hint">Enter an already available '
                    'fictional finding, or simulate obtaining another test.</p>'
                )
                manual_test = gr.Dropdown(
                    choices=[(label, key) for key, (label, _) in TESTS.items()],
                    value="cbc",
                    label="Diagnostic test",
                    elem_id="nq-manual-test",
                )
                result = gr.Radio(
                    ["Not flagged", "Abnormal / flagged"],
                    value="Not flagged",
                    label="Fictional result",
                    elem_id="nq-finding",
                )
                existing = gr.Checkbox(
                    value=True,
                    label="Result already available (no extra effort)",
                    elem_id="nq-existing",
                )
                btn_manual = gr.Button(
                    "Record fictional result",
                    interactive=False,
                    variant="secondary",
                    elem_id="nq-record",
                )

        with gr.Column(
            scale=7, min_width=350, elem_classes=["nq-insights-column"]
        ):
            gr.HTML(
                '<div class="nq-step-title"><span>03</span> Diagnostic insight</div>'
            )
            with gr.Row(equal_height=False, elem_classes=["nq-metric-row"]):
                status = gr.HTML(value=_blank("Assessment overview", "Choose a fictional presentation and start the assessment."))
                burden = gr.HTML(value=_blank("Testing burden", "Simulated acquisition effort will be tracked here."))
            differential = gr.HTML(value=_blank("Possible explanations", "Model-ranked conditions will appear after intake."))
            tests = gr.HTML(value=_blank("Next informative tests", "The model will rank unobserved tests after assessment."))
            pathway = gr.HTML(value=_blank("Investigation timeline", "Each revealed test finding will be documented here."))

    gr.HTML(
        '<p class="nq-foot"><strong>NEXTQUESTION</strong> · Abdulla Alhajaj, '
        'Zayed Alketbi, Mohammed Alali · Veritas AI 2026. This application '
        'contains only simplified, synthetic condition labels, test results '
        'and effort points. Its keyword parser, clinical predictions, '
        'and urgency warnings are not validated for real patients. '
        '<strong>This project is a prototype and is not medical advice.</strong></p>'
    )

    state = gr.State(None)
    outputs = [
        state, status, differential, tests, pathway,
        burden, btn_next, btn_manual,
    ]

    # Changing patient intake must always invalidate the earlier assessment.
    scenario.change(
        case_input, inputs=scenario, outputs=[complaint, selected]
    ).then(reset, outputs=outputs, show_progress="hidden")
    mode.change(
        lambda m: (
            gr.update(interactive=m == SIMULATED_CASE),
            gr.update(visible=m == SIMULATED_CASE),
            gr.update(visible=m == MANUAL_CASE),
        ),
        inputs=mode,
        outputs=[scenario, simulated_panel, manual_panel],
    ).then(reset, outputs=outputs, show_progress="hidden")
    complaint.change(reset, outputs=outputs, show_progress="hidden")
    selected.change(reset, outputs=outputs, show_progress="hidden")
    flags.change(reset, outputs=outputs, show_progress="hidden")
    btn_start.click(
        start, inputs=[mode, scenario, complaint, selected, flags],
        outputs=outputs, show_progress="minimal",
    )
    btn_next.click(
        acquire, inputs=state, outputs=outputs, show_progress="minimal",
    )
    btn_manual.click(
        record_result, inputs=[state, result, manual_test, existing],
        outputs=outputs, show_progress="minimal",
    )
    btn_reset.click(reset, outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        theme=THEME,
        css=CSS,
    )
