"""Separate public MIMIC-IV hospital-record replay (RESEARCH ONLY).

Real de-identified public data, not simulated lab results. Retrospective:
the candidate pool consists of labs actually performed historically,
so ranking has hindsight availability bias and is NOT a prospective
clinical test recommendation. No clinical safety or cost benefit established.
"""
from __future__ import annotations
from html import escape
from pathlib import Path
import gradio as gr
import numpy as np
from hospital.mimic_real import MAX_STEPS, load_research

ART = Path("artifacts/mimic_real")
READY = all((ART / name).exists() for name in
            ("real_model.json","real_demo_cases.json","real_metrics.json"))
MODEL = CASES = METRICS = LAB_NAMES = None
if READY:
    try:
        MODEL, CASES, METRICS, LAB_NAMES = load_research(str(ART))
    except (OSError,ValueError,KeyError,TypeError):
        READY = False

def _panel(title,text):
    return ('<div class="nq-card"><h3>'+escape(str(title))+'</h3>'
            '<p class="nq-muted">'+escape(str(text))+'</p></div>')

def _render(state):
    if not state:
        return (
            _panel("Real-record assessment","Select a de-identified validation encounter and start."),
            _panel("Model-ranked discharge categories","Scores shown here are not medical diagnoses."),
            _panel("Historically observed lab indicators","Only recorded findings can be replayed."),
            _panel("Recorded lab findings","No results revealed yet."),
            gr.update(interactive=False)
        )
    case=CASES[state["index"]]
    known=state["obtained"]
    p=MODEL.posterior(case["complaint"],known)
    top=np.argsort(-p)[:5]
    # Safety: even the largest research posterior is not a clinical diagnosis.
    # This is a descriptive display threshold, not a clinical triage threshold.
    uncertain = float(np.max(p)) < 0.50
    remaining=MODEL.rank(case["complaint"],known,eligible=set(case["available_labs"]))
    may_continue=bool(remaining) and len(known)<MAX_STEPS
    header=(
        '<div class="nq-card"><h3>De-identified public encounter</h3>'
        '<div class="nq-line"><span>Scenario</span><strong>'+escape(case["id"])+'</strong></div>'
        '<div class="nq-line"><span>Recorded chief complaint</span><strong>'+
            escape(case["complaint"] or "Not available")+'</strong></div>'
        f'<div class="nq-line"><span>Recorded labs revealed</span><span class="nq-pill">{len(known)} of {len(case["available_labs"])}</span></div>'
        '<p class="nq-muted">All model training used different patients; no IDs or dates are displayed.</p>'
        + (('<div class="nq-line"><span>Historical primary discharge group</span><strong>'+
           escape(case["label"])+'</strong></div>') if not may_continue else '')
        + '</div>'
    )
    differential=['<div class="nq-card"><h3>Model-ranked discharge categories</h3>']
    if uncertain:
        differential.append(
            '<div class="nq-badge">High uncertainty in this small research sample. '
            'No diagnosis is established; do not use this ranking for medical decisions.</div>'
        )
    for index in top:
        idx=int(index)
        value=float(p[idx])
        differential.append(
            '<div class="nq-line"><div style="flex:1;min-width:160px"><strong>'+
            escape(MODEL.labels[idx])+'</strong><div class="nq-bar"><span style="width:'+
            f'{100*value:.1f}%"></span></div></div><span class="nq-pill">{value:.1%}</span></div>'
        )
    differential.append('<p class="nq-muted">These are exploratory model weights, not calibrated clinical diagnoses.</p></div>')
    ranked=['<div class="nq-card"><h3>Ranked historical laboratory findings</h3>']
    for n,entry in enumerate(remaining[:4],1):
        name=LAB_NAMES.get(entry["lab"],entry["lab"])
        ranked.append(
            f'<div class="nq-line"><strong>{n}. {escape(name)}</strong>'
            f'<span class="nq-pill">{entry["info_gain"]:.3f} nats</span></div>'
        )
    if not remaining:
        ranked.append('<p class="nq-muted">No remaining recorded test findings.</p>')
    ranked.append(
        '<p class="nq-muted">IMPORTANT: the pool of historically observed labs is '
        'known in hindsight; results of unperformed tests do not exist. '
        'This replay cannot prove which test would have been needed.</p></div>'
    )
    history=['<div class="nq-card"><h3>Actual recorded lab flags</h3>']
    for entry in state["history"]:
        history.append('<div class="nq-step"><strong>'+
            escape(LAB_NAMES.get(entry["lab"],entry["lab"]))+
            '</strong><p class="nq-muted">'+
            ('Flagged abnormal' if entry["flagged"] else 'No abnormal flag recorded')+
            '</p></div>')
    if not state["history"]:
        history.append('<p class="nq-muted">No lab outcomes revealed.</p>')
    history.append('</div>')
    return header,''.join(differential),''.join(ranked),''.join(history),gr.update(interactive=may_continue)

def real_start(case_id):
    if not READY or not CASES:
        return None,*_render(None)
    index=next((i for i,c in enumerate(CASES) if c["id"]==case_id),None)
    if index is None:
        return None,*_render(None)
    state={"index":index,"obtained":{},"history":[]}
    return state,*_render(state)

def real_next(state):
    if not READY or not state or len(state["obtained"])>=MAX_STEPS:
        return state,*_render(state)
    case=CASES[state["index"]]
    ranked=MODEL.rank(case["complaint"],state["obtained"],eligible=set(case["available_labs"]))
    if not ranked:
        return state,*_render(state)
    lab=ranked[0]["lab"]
    # The algorithm cannot read this result before deciding on the lab.
    flag=int(case["available_labs"][lab])
    updated={"index":state["index"],"obtained":dict(state["obtained"]),
             "history":list(state["history"])}
    updated["obtained"][lab]=flag
    updated["history"].append({"lab":lab,"flagged":flag})
    return updated,*_render(updated)

def real_reset():
    return None,*_render(None)

def build_pilot():
    """Mount inside an active gr.Blocks component tree."""
    if not READY:
        gr.HTML(_panel("Public hospital pilot not prepared",
            "Run python -m hospital.mimic_real with public internet access. "
            "No fake or substitute real lab observations will be supplied."))
        return
    a=METRICS["adaptive"];b=METRICS["fixed"]
    gr.HTML(
        '<div class="nq-benchmark">'
        '<div class="nq-bench-tile"><span class="nq-bench-label">Held-out admissions</span>'
        f'<span class="nq-bench-value">{a["n_admissions"]}</span></div>'
        '<div class="nq-bench-tile"><span class="nq-bench-label">Adaptive accuracy</span>'
        f'<span class="nq-bench-value">{a["top1_accuracy"]:.1%}</span></div>'
        '<div class="nq-bench-tile"><span class="nq-bench-label">Fixed-order accuracy</span>'
        f'<span class="nq-bench-value">{b["top1_accuracy"]:.1%}</span></div>'
        '<div class="nq-bench-tile"><span class="nq-bench-label">Both approaches</span>'
        f'<span class="nq-bench-value">{a["mean_recorded_lab_indicators_queried"]:.1f}</span>'
        '<small>historically recorded lab flags per admission</small></div>'
        '<p class="nq-benchmark-foot">Adaptive was WORSE than fixed ordering on '
        'this tiny real-record sample. This is not a patient-care validation.</p></div>'
    )
    gr.HTML(
        '<div class="nq-safety"><strong>Real-record research only:</strong> '
        'MIMIC-IV and MIMIC-IV-ED Demo v2.2, 100 de-identified patients '
        '(ODbL 1.0). Not every record has a complaint. Only historically '
        'performed tests can be replayed. Availability is known in hindsight '
        'and results have selection bias. No validated diagnosis, test necessity '
        'or medical cost claims. This project is a prototype and is not medical advice.</div>'
    )
    cases=[(x["id"]+" · "+x["complaint"][:42],x["id"]) for x in CASES]
    pick=gr.Dropdown(choices=cases,value=CASES[0]["id"],
                     label="Public validation encounter",elem_id="nq-real-case")
    with gr.Row():
        begin=gr.Button("Explore public encounter",variant="primary")
        next_button=gr.Button("Reveal recorded lab finding",interactive=False)
        reset_button=gr.Button("Reset pilot",variant="secondary")
    with gr.Row():
        with gr.Column():
            overview=gr.HTML(value=_panel("Real-record assessment","Choose an encounter."))
            history=gr.HTML(value=_panel("Historical lab findings","No result yet."))
        with gr.Column():
            categories=gr.HTML(value=_panel("Predicted discharge categories","No prediction yet."))
            ranking=gr.HTML(value=_panel("Remaining lab information","No ranking yet."))
    state=gr.State(None)
    outputs=[state,overview,categories,ranking,history,next_button]
    begin.click(real_start,inputs=pick,outputs=outputs)
    next_button.click(real_next,inputs=state,outputs=outputs)
    pick.change(real_reset,outputs=outputs)
    reset_button.click(real_reset,outputs=outputs)
