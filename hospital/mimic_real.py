"""Public REAL hospital EHR pilot from MIMIC-IV Clinical + ED Demo v2.2.

No data is manufactured: encounter complaints, the primary discharge-diagnosis
chapter, and lab results are obtained from PhysioNet's 100-person public demo.

IMPORTANT LIMITATIONS:
* Only 100 patients: exploratory, NOT a diagnostic model for clinical use.
* Discharge ICD chapter is a noisy coarse proxy, not verified diagnosis truth.
* Only labs observed in the first 24 hours are eligible for replay, an
  optimistic retrospective selection bias; unperformed tests are UNKNOWN.
* A missing lab is never imputed as "normal" and no counterfactual test result
  is invented. "Flagged" means the source's abnormal flag, not severity.
* No actual prices, patient savings or clinical necessity labels.
* Patient-level splits keep all admissions of an individual in one partition.
* No raw identifiers are exported with the judge-facing artifact.
Source: Johnson et al, MIMIC-IV Clinical Database Demo 2.2, PhysioNet,
DOI:10.13026/dp1f-ex47; MIMIC-IV-ED Demo 2.2 DOI:10.13026/jzz5-vs76.
Both datasets: Open Data Commons Open Database License v1.0.
"""
from __future__ import annotations
import collections
import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np
from hospital.mimic_probe import chapter, get, get_ed

SEED = 22881
# Defined before observing the final evaluation partition. Each item is an
# actual MIMIC d_labitems itemid and records its actual flagged/normal result.
LAB_CODES = (
    "50931", # glucose
    "51222", # hemoglobin
    "51279", # red blood cell count
    "51301", # white blood cell count
    "50912", # creatinine
    "51006", # urea nitrogen
    "51265", # platelets
    "50971", # potassium
    "50983", # sodium
    "51237", # INR(PT)
    "50882", # bicarbonate
    "50970", # phosphate
)
GROUPS = ("Circulatory","Digestive","Endocrine/metabolic",
          "Genitourinary","Infectious","Neoplasm","Other","Respiratory")
MAX_STEPS = 4
RESEARCH_DISCLAIMER = ("REAL de-identified public hospital records (100-person demo). "
    "Retrospective lab findings and coarse ICD discharge chapters, not patient "
    "diagnoses or valid prospective test recommendations. No clinical or monetary claims. "
    "This project is a prototype and is not medical advice.")


def _date(text):
    try: return dt.datetime.fromisoformat(text)
    except (ValueError, TypeError): return None


def _words(complaint):
    # Coarse, non-sensitive text features. Do not use patient names or IDs.
    return set(re.findall(r"[a-z]{3,}", str(complaint).lower())) - {
        "the","for","and","with","from","this","that","since",
        "eval","evaluation","other","patient","followup",
    }


def load_cases():
    admission_rows = get("admissions")
    diagnosis_rows = get("diagnoses_icd")
    events = get("labevents")
    lab_names = {
        row["itemid"]: row["label"] for row in get("d_labitems")
    }
    ed = get_ed("edstays")
    triage = {row["stay_id"]:row for row in get_ed("triage")}
    chief = {}
    for row in ed:
        hadm = row.get("hadm_id","")
        item = triage.get(row.get("stay_id",""),{})
        if hadm and item.get("chiefcomplaint",""):
            chief.setdefault(hadm, item["chiefcomplaint"])
    primary = {}
    for row in diagnosis_rows:
        hadm = row.get("hadm_id","")
        if not hadm: continue
        seq = int(row["seq_num"]) if row["seq_num"].isdigit() else 100000
        if hadm not in primary or seq < primary[hadm][0]:
            primary[hadm] = (seq,chapter(row))
    by_adm = {}
    for row in admission_rows:
        hadm = row.get("hadm_id","")
        if hadm not in primary or primary[hadm][1] not in GROUPS:
            continue
        date = _date(row["admittime"])
        if not date: continue
        by_adm[hadm] = {
            "hadm":hadm,"patient":row["subject_id"],"start":date,
            "group":primary[hadm][1],
            "complaint": chief.get(hadm, ""),
            "known": {},
            "admission_type":row.get("admission_type","")
        }
    for r in events:
        hadm = r.get("hadm_id","")
        if hadm not in by_adm or r.get("itemid","") not in LAB_CODES:
            continue
        c = by_adm[hadm]
        date = _date(r.get("charttime",""))
        # Only *initial 24h* results. Anything at discharge or later is excluded.
        if not date or not (c["start"] <= date <= c["start"] + dt.timedelta(hours=24)):
            continue
        if not r.get("valuenum","").strip():
            continue
        try:
            numeric = float(r["valuenum"])
        except (TypeError,ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        item = r["itemid"]
        prev = c["known"].get(item)
        # Earliest recorded measurement per admission and lab item.
        if prev is None or date < prev[0]:
            c["known"][item] = (date, int(r.get("flag","").strip().lower()=="abnormal"))
    episodes = []
    for c in by_adm.values():
        observed = {k:v[1] for k,v in c["known"].items()}
        if len(observed) < 2:
            continue
        episodes.append({
            "patient":c["patient"],"hadm":c["hadm"],"group":c["group"],
            "complaint":c["complaint"],"admission_type":c["admission_type"],
            "observed":observed,
        })
    # Never export identifiers: patient/hadm are only internal split keys.
    return episodes, {k:lab_names.get(k,k) for k in LAB_CODES}, {
        "downloaded_public_patients":100, "public_admission_rows":len(admission_rows),
        "selected_episodes":len(episodes),
        "selected_distinct_patients":len({r["patient"] for r in episodes}),
        "ed_complaints_linked":sum(bool(r["complaint"]) for r in episodes),
        "label_distribution":dict(collections.Counter(r["group"] for r in episodes)),
    }


def patient_split(rows,seed=SEED):
    people=sorted(set(r["patient"] for r in rows))
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(people))
    t=int(len(order)*.65);v=int(len(order)*.80)
    groups=[set(people[i] for i in part) for part in (order[:t],order[t:v],order[v:])]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    split=[[r for r in rows if r["patient"] in people_set] for people_set in groups]
    assert sum(len(s) for s in split)==len(rows)
    return split


class RealEvidenceModel:
    def __init__(self,train, labels=None):
        self.labels=tuple(labels or GROUPS)
        self.index={name:j for j,name in enumerate(self.labels)}
        self.lab_codes=LAB_CODES
        self.vocab=tuple(sorted(
            set().union(*(_words(x["complaint"]) for x in train))
        )) if train else ()
        counts=np.ones(len(self.labels),dtype=float)
        npos=np.ones((len(self.labels),len(LAB_CODES)),dtype=float)
        nobserved=np.full((len(self.labels),len(LAB_CODES)),2.,dtype=float)
        wordpos=np.ones((len(self.labels),len(self.vocab)),dtype=float)
        wi={w:i for i,w in enumerate(self.vocab)}
        for row in train:
            j=self.index[row["group"]]
            counts[j]+=1
            for word in _words(row["complaint"]):
                if word in wi: wordpos[j,wi[word]]+=1
            for k,outcome in row["observed"].items():
                if k in LAB_CODES:
                    p=LAB_CODES.index(k)
                    nobserved[j,p]+=1
                    npos[j,p]+=int(outcome)
        self.prior=counts/counts.sum()
        self.p_lab=npos/nobserved
        # Bernoulli smoothing for complaint features, NOT measured symptoms.
        self.p_word=wordpos/(counts[:,None]+1)
        self.wi=wi

    def posterior(self,complaint="",obtained=None):
        obtained=obtained or {}
        if not set(obtained).issubset(LAB_CODES):
            raise ValueError("Unknown laboratory indicator")
        if any(x not in (0,1) for x in obtained.values()):
            raise ValueError("Expected 0/1 recorded abnormal flag")
        score=np.log(self.prior.copy())
        for word in sorted(_words(complaint)):
            if word in self.wi:
                score+=np.log(np.clip(self.p_word[:,self.wi[word]],1e-8,1))
        for lab,outcome in sorted(obtained.items()):
            p=self.p_lab[:,LAB_CODES.index(lab)]
            score+=np.log(p if outcome else 1-p)
        score-=score.max()
        probs=np.exp(score);return probs/probs.sum()

    def rank(self,complaint="",obtained=None,eligible=None):
        obtained=obtained or {}
        p=self.posterior(complaint,obtained)
        base=-float(np.sum(p*np.log(np.clip(p,1e-12,1))))
        ranked=[]
        for j,lab in enumerate(LAB_CODES):
            if lab in obtained or (eligible is not None and lab not in eligible):
                continue
            q=self.p_lab[:,j]
            prob_pos=float(p@q)
            pos=p*q;neg=p*(1-q)
            pos/=pos.sum();neg/=neg.sum()
            ent1=-float(np.sum(pos*np.log(np.clip(pos,1e-12,1))))
            ent0=-float(np.sum(neg*np.log(np.clip(neg,1e-12,1))))
            gain=max(0.,base-(prob_pos*ent1+(1-prob_pos)*ent0))
            ranked.append({"lab":lab,"info_gain":gain})
        return sorted(ranked,key=lambda x:(-x["info_gain"],x["lab"]))


def replay(model, case, policy="adaptive",max_tests=MAX_STEPS):
    """Retrospective-only: candidate pool restricted to truly recorded labs.

    Not a prospective policy; selecting from observed lab availability is
    subject to care-selection bias. Outcomes remain hidden until selected.
    """
    obtained={}
    chosen=[]
    eligible=set(case["observed"])
    for _ in range(max_tests):
        if policy=="adaptive":
            ranks=model.rank(case["complaint"],obtained,eligible)
            key=ranks[0]["lab"] if ranks else None
        elif policy=="fixed":
            choices=[x for x in LAB_CODES if x in eligible and x not in obtained]
            key=choices[0] if choices else None
        else:
            raise ValueError("Unknown benchmark policy")
        if key is None:break
        obtained[key]=case["observed"][key]
        chosen.append(key)
    probabilities=model.posterior(case["complaint"],obtained)
    label=model.labels[int(np.argmax(probabilities))]
    return {"actual":case["group"],"predicted":label,
            "correct":int(label==case["group"]),"queried":chosen,
            "probability":float(probabilities.max()),"number":len(chosen)}


def evaluate(model, cases, policy):
    if not cases:raise ValueError("No evaluation cases")
    paths=[replay(model,c,policy) for c in cases]
    confusion=[[sum(r["actual"]==a and r["predicted"]==b for r in paths)
                for b in GROUPS] for a in GROUPS]
    return {
        "n_admissions":len(paths),
        "top1_accuracy":sum(r["correct"] for r in paths)/len(paths),
        "mean_recorded_lab_indicators_queried":sum(r["number"] for r in paths)/len(paths),
        "labels":list(GROUPS),
        "confusion_matrix_actual_rows_predicted_columns":confusion,
        "misclassifications":len(paths)-sum(r["correct"] for r in paths),
    }


def export_research(outdir="artifacts/mimic_real"):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    records, labnames, stats=load_cases()
    train,val,test=patient_split(records)
    assert len(train)>30 and len(test)>5, "Too few linked real episodes to assess"
    model=RealEvidenceModel(train)
    adaptive=evaluate(model,test,"adaptive")
    fixed=evaluate(model,test,"fixed")
    # Validation-only cases for the actual site. No subject_id, hadm_id exported.
    examples=[]
    for i,c in enumerate(val[:12]):
        examples.append({
            "id":f"Public EHR example {i+1:02d}",
            "complaint":c["complaint"] or "No linked ED chief complaint",
            "available_labs":c["observed"],"label":c["group"],
        })
    report={
        "title":"REAL MIMIC-IV public demo: retrospective preliminary pilot",
        "source":"PhysioNet MIMIC-IV Clinical Database Demo 2.2 + MIMIC-IV-ED Demo 2.2",
        "citations":[
            "https://doi.org/10.13026/dp1f-ex47",
            "https://doi.org/10.13026/jzz5-vs76",
        ],
        "license":"ODbL 1.0",
        "real_deidentified_hospital_records":True,
        "patient_level_split":True,
        "split_admissions":{"train":len(train),"validation":len(val),"test":len(test)},
        "split_patients":{"train":len({r["patient"] for r in train}),"validation":len({r["patient"] for r in val}),"test":len({r["patient"] for r in test})},
        "coverage":stats,
        "n_laboratory_indicators":len(LAB_CODES),
        "findings":"Real MIMIC first-24h lab abnormal flags, not fabricated outcomes",
        "target":"Primary ICD-9/10 discharge diagnosis GROUP (not verified actual differential)",
        "adaptive":adaptive,"fixed":fixed,
        "demo_validation_only":True,
        "safety":RESEARCH_DISCLAIMER,
        "important_limitations":[
            "100-person sample is tiny and biased; findings not externally validated",
            "Many admissions have no ED chief complaint; recorded text is incomplete",
            "No ability to know results of a test that was not ordered historically",
            "Replay restricts the candidate pool to historically observed labs: selection bias",
            "All observed tests are counted as one item, not actual blood draws or charges",
            "Primary discharge ICD group is an imperfect retrospective proxy",
            "No real costs, unnecessary-test labels, or prospective diagnostic effects",
            "Missing lab result is unknown, not a negative finding",
        ],
    }
    (out/"real_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"real_demo_cases.json").write_text(json.dumps(examples,indent=2),encoding="utf-8")
    params={
        "prior":model.prior.tolist(),"p_lab":model.p_lab.tolist(),
        "p_word":model.p_word.tolist(),"labels":list(model.labels),
        "vocab":list(model.vocab),"labnames":labnames,
    }
    (out/"real_model.json").write_text(json.dumps(params),encoding="utf-8")
    print("REAL PATIENT-LEVEL SPLIT",report["split_patients"],report["split_admissions"])
    print("REAL RETROSPECTIVE PILOT",json.dumps({
        "adaptive":adaptive,"fixed":fixed,
    },indent=2))
    return report


def load_research(path="artifacts/mimic_real"):
    out=Path(path)
    raw=json.loads((out/"real_model.json").read_text())
    model=RealEvidenceModel.__new__(RealEvidenceModel)
    model.labels=tuple(raw["labels"]);model.index={x:i for i,x in enumerate(model.labels)}
    model.lab_codes=LAB_CODES;model.vocab=tuple(raw["vocab"])
    model.wi={w:i for i,w in enumerate(model.vocab)}
    model.prior=np.array(raw["prior"]);model.p_lab=np.array(raw["p_lab"])
    model.p_word=np.array(raw["p_word"])
    examples=json.loads((out/"real_demo_cases.json").read_text())
    metrics=json.loads((out/"real_metrics.json").read_text())
    return model,examples,metrics,raw["labnames"]


if __name__=="__main__":
    export_research()
