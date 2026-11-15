"""Inspect the REAL, publicly licensed MIMIC-IV Demo v2.2 EHR subset.

The 100-person PhysioNet sample contains genuinely de-identified hospital
records, not our own generated symptom/test outcomes. This script is a first
data-quality gate, not a trained clinical model.
Source DOI: https://doi.org/10.13026/dp1f-ex47 (ODbL 1.0).
"""
import collections
import csv
import gzip
import io
import json
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/mimic-iv-demo/2.2/hosp/"
FILES = ["patients", "admissions", "diagnoses_icd", "d_labitems", "labevents"]


def get(name, cache=Path("artifacts/mimic_real/source")):
    cache.mkdir(parents=True, exist_ok=True)
    filename = cache / (name + ".csv.gz")
    if not filename.is_file():
        req = urllib.request.Request(BASE + name + ".csv.gz", headers={"User-Agent":"NEXTQUESTION-Veritas-Academic/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        if not content.startswith(b"\x1f\x8b"):
            raise ValueError(f"Remote MIMIC download was not gzip: {name}")
        filename.write_bytes(content)
    with gzip.open(filename, mode="rt", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def chapter(row):
    code = row["icd_code"]
    version = row["icd_version"]
    if version == "10":
        first = code[:1].upper()
        return {
            "I":"Circulatory", "J":"Respiratory", "K":"Digestive",
            "N":"Genitourinary", "E":"Endocrine/metabolic",
            "A":"Infectious", "B":"Infectious",
            "G":"Neurologic", "M":"Musculoskeletal",
            "D":"Blood/immune", "C":"Neoplasm"
        }.get(first, "Other")
    if version == "9" and code and code[0].isdigit():
        n = int(code[:3])
        for low, high, value in [
            (1, 139, "Infectious"),(140, 239, "Neoplasm"),
            (240,279,"Endocrine/metabolic"),(280,289,"Blood/immune"),
            (320,389,"Neurologic"),(390,459,"Circulatory"),
            (460,519,"Respiratory"),(520,579,"Digestive"),
            (580,629,"Genitourinary"),(710,739,"Musculoskeletal")
        ]:
            if low <= n <= high: return value
    return "Other"


def main():
    tables = {name:get(name) for name in FILES}
    print("MIMIC PUBLIC REAL TABLE SIZES", {k:len(v) for k,v in tables.items()})
    print("LAB DATA COLUMNS",list(tables["labevents"][0]))
    print("DIAGNOSIS COLUMNS",list(tables["diagnoses_icd"][0]))
    print("ADMISSIONS COLUMNS",list(tables["admissions"][0]))
    by_hadm = collections.defaultdict(list)
    for row in tables["diagnoses_icd"]:
        if row["hadm_id"]:
            by_hadm[row["hadm_id"]].append(row)
    diagnoses = {}
    for hadm, entries in by_hadm.items():
        main = min(entries,key=lambda v:int(v["seq_num"]) if v["seq_num"].isdigit() else 99999)
        diagnoses[hadm] = chapter(main)
    print("PRIMARY DIAGNOSIS GROUPS",dict(collections.Counter(diagnoses.values())))
    measured = collections.defaultdict(set)
    abnormal = collections.Counter()
    by_lab = collections.Counter()
    with_hadm=0
    for row in tables["labevents"]:
        hadm=row["hadm_id"]
        if not hadm or hadm not in diagnoses: continue
        with_hadm+=1
        item=row["itemid"]
        if row["flag"].strip().lower() not in ("normal","abnormal"): continue
        measured[item].add(hadm)
        if row["flag"].strip().lower()=="abnormal": abnormal[item]+=1
        by_lab[item]+=1
    labels={row["itemid"]:row["label"] for row in tables["d_labitems"]}
    top=sorted(measured, key=lambda k:len(measured[k]), reverse=True)[:28]
    top_labs=[{"itemid":k,"label":labels.get(k,""),
      "distinct_admissions":len(measured[k]),"readings":by_lab[k],"abnormal":abnormal[k]} for k in top]
    print("LABS WITH HADM",with_hadm)
    print("TOP RECORDED LAB INDICATORS",json.dumps(top_labs,indent=2))
    summary={"source":"MIMIC-IV Clinical Database Demo v2.2, PhysioNet; real, deidentified 100-person records", "counts":{k:len(v) for k,v in tables.items()},"admission_primary_groups":dict(collections.Counter(diagnoses.values())),"top_labs":top_labs}
    path=Path("artifacts/mimic_real/probe.json");path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    main()
