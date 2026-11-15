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

CSS = """
.gradio-container { max-width: 1240px !important; margin: auto !important;
                    background: #09111e !important; color: #e8f3fa !important; }
.gradio-container, .gradio-container * { box-sizing: border-box; }
.nq-hero { background: linear-gradient(115deg,#111f34,#142a37); padding: 26px 28px;
           border: 1px solid #304557; border-radius: 18px; margin: 14px 0; }
.nq-kicker { color: #74ebc2 !important; letter-spacing: .17em; font-size: 12px; font-weight: 700; }
.nq-hero h1 { color: #f5fbff !important; margin: 8px 0; font-size: clamp(27px,4vw,39px); }
.nq-hero p, .nq-p { color: #c7d9e3 !important; line-height: 1.55; margin: 4px 0; }
.nq-stats { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
.nq-stat { padding:8px 12px; border-radius:9px; background:#203447; color:#ddf7ff !important; font-size:12px; }
.nq-card { background: #152637; border: 1px solid #345065; border-radius: 14px;
           padding: 17px 18px; margin: 10px 0; }
.nq-card *, .nq-card p, .nq-card div, .nq-card span { color: #deedf6; opacity:1 !important; }
.nq-card h3 { margin:0 0 11px; color:#ffffff !important; font-size:17px; }
.nq-card .nq-muted { font-size:12px; color:#bcd3df !important; }
.nq-card .nq-emphasis { color: #7becce !important; font-weight:800; }
.nq-card .nq-warning { color: #ffe7a0 !important; font-weight: 700; }
.nq-line { display:flex; justify-content:space-between; flex-wrap:wrap; align-items:center;
           gap: 10px; padding:10px 0; border-bottom:1px solid #395164; font-size:13px; }
.nq-line:last-child { border-bottom:0; }
.nq-pill { background:#244253; border:1px solid #426579; padding:4px 8px;
           border-radius:20px; color:#d4f8f0 !important; font-size:11px; white-space:nowrap; }
.nq-bar { height: 7px; background: #304353; border-radius: 12px; overflow: hidden; margin:7px 0; }
.nq-bar > span { display:block; height:100%; background:#6ceac5; }
.nq-step { border-left:2px solid #76e8cd; padding:10px 15px; margin:8px 0; }
.nq-step strong { color:#7df6d5 !important; }
.nq-badge { background:#3a3026; border:1px solid #a9844c; color:#ffe4ae !important;
            border-radius:8px; padding:7px 10px; margin:9px 0; }
.nq-metric { color:#ffffff !important; font-size:30px; font-weight:850; }
.nq-foot { font-size:12px; color:#d8e7f0 !important; line-height:1.55; margin:18px 0; }
@media(max-width:650px) {
  .nq-hero { padding:17px; }
  .nq-card { padding:12px; }
  .nq-line { align-items:flex-start; }
  .nq-stat { flex:1; min-width:120px; }
}
"""
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
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
    if ranking:
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
            f'<p>{escape(finding)} · +{COSTS[step["key"]]} simulated pts '
            f'· leading model weight {step["confidence"]:.1%}</p></div>'
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
    enabled = bool(ranking) and len(known) < 6
    return status, differential, tests, timeline, burden, (
        gr.update(interactive=enabled and state["mode"] == SIMULATED_CASE)
    ), (
        gr.update(interactive=enabled and state["mode"] == MANUAL_CASE)
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


def _add_result(state, finding, test_key=None):
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
    state = {**state, "obtained": dict(state["obtained"]),
             "history": list(state["history"])}
    state["obtained"][key] = finding
    state["cost"] += COSTS[key]
    confidence = float(MODEL.posterior(state["symptoms"], state["obtained"]).max())
    state["history"].append({"key": key, "value": finding, "confidence": confidence})
    return state, *_render(state)


def acquire(state):
    if not state or state.get("mode") != SIMULATED_CASE:
        return state, *_render(state)
    case = BY_SCENARIO[state["scenario"]]
    ranked = MODEL.ranking(state["symptoms"], state["obtained"],
                           state["budget"] - state["cost"])
    if not ranked:
        return state, *_render(state)
    key = ranked[0]["key"]
    return _add_result(state, int(case["test_findings"][key]))


def record_result(state, result, test_key=None):
    if not state or state.get("mode") != MANUAL_CASE:
        return state, *_render(state)
    return _add_result(state, 1 if result == "Abnormal / flagged" else 0, test_key)


def reset():
    return None, *_render(None)


with gr.Blocks(title="NEXTQUESTION | Hospital diagnostic test prioritization") as demo:
    gr.HTML(f"""
    <section class="nq-hero">
      <div class="nq-kicker">NEXTQUESTION · AI + HEALTH · RESEARCH PROTOTYPE</div>
      <h1>Better questions. Fewer unnecessary investigations?</h1>
      <p>Explore how an adaptive AI could help clinicians choose which diagnostic
      information to collect next — across different presenting complaints.</p>
      <div class="nq-stats">
        <span class="nq-stat">{len(CONDITIONS)} simulated conditions</span>
        <span class="nq-stat">{DEPT_COUNT} clinical categories</span>
        <span class="nq-stat">{len(TESTS)} illustrative test types</span>
        <span class="nq-stat">No patient data</span>
      </div>
    </section>
    """)
    gr.HTML(
        '<div class="nq-badge"><strong>Safety:</strong> This is an artificial educational '
        'simulation, NOT a diagnostic or medical device. Never enter real patient '
        'information or use suggestions to delay care, refuse tests, or make medical decisions.</div>'
    )
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 01 / Patient presentation")
            mode = gr.Radio(
                choices=[SIMULATED_CASE, MANUAL_CASE], value=SIMULATED_CASE,
                label="Research mode",
            )
            scenario = gr.Dropdown(
                choices=list(BY_SCENARIO),
                value=SCENARIOS[0]["id"],
                label="Synthetic example (covers multiple medical categories)",
            )
            complaint = gr.Textbox(
                label="Presenting complaint (fictional free text)",
                value=SCENARIOS[0]["complaint"],
                placeholder="Example: fever, cough and trouble breathing",
                lines=2,
            )
            selected = gr.CheckboxGroup(
                choices=[(label, key) for key, label in SYMPTOMS.items()],
                value=SCENARIOS[0]["symptoms"],
                label="Reported symptom signals · unselected = unknown",
            )
            flags = gr.CheckboxGroup(
                choices=[(label, key) for key, label in RED_FLAGS.items()],
                label="Safety red flags (fictional case only)",
            )
            with gr.Row():
                btn_start = gr.Button("Start assessment", variant="primary")
                btn_reset = gr.Button("Reset", variant="secondary")
            gr.Markdown("### 02 / Results")
            with gr.Row():
                btn_next = gr.Button("Simulate suggested test result", interactive=False)
                result = gr.Radio(
                    ["Not flagged", "Abnormal / flagged"],
                    value="Not flagged",
                    label="Manual fictional finding",
                )
                manual_test = gr.Dropdown(
                    choices=[(label, key) for key, (label, _) in TESTS.items()],
                    value="cbc",
                    label="Known or clinician-chosen fictional test",
                )
                btn_manual = gr.Button("Record entered result", interactive=False)
            summary = gr.HTML()
            burden = gr.HTML()
        with gr.Column(scale=6):
            gr.Markdown("### 03 / AI reasoning")
            status = gr.HTML()
            differential = gr.HTML()
            tests = gr.HTML()
            pathway = gr.HTML()
    gr.HTML(
        '<p class="nq-foot">NEXTQUESTION · Abdulla Alhajaj, Zayed Alketbi, Mohammed Alali'
        ' · Veritas AI International High School Hackathon 2026. '
        'Symptom extraction is keyword-based; all conditions, test outcomes and '
        'cost units in this demonstration are synthetic. A licensed clinician '
        'must order or interpret any real diagnostic test. Not medical advice.</p>'
    )
    state = gr.State(None)
    outputs = [state, status, differential, tests, pathway, burden, btn_next, btn_manual]
    scenario.change(case_input, inputs=scenario, outputs=[complaint, selected])
    mode.change(
        lambda m: gr.update(interactive=m == SIMULATED_CASE),
        inputs=mode, outputs=scenario,
    )
    btn_start.click(start, inputs=[mode, scenario, complaint, selected, flags],
                    outputs=outputs, show_progress="minimal")
    btn_next.click(acquire, inputs=state, outputs=outputs, show_progress="minimal")
    btn_manual.click(record_result, inputs=[state, result, manual_test], outputs=outputs,
                     show_progress="minimal")
    btn_reset.click(reset, outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        theme=THEME,
        css=CSS,
    )
