# make_batch_csv.py
import pandas as pd, json
from pathlib import Path
from datetime import date

DATA_ROOT = Path("path/to/root")  # contains patients/, conditions/, observations/
OUT = Path("examples/clinical_batch_examples.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

patients   = pd.read_csv(DATA_ROOT/"patients"/"patients.csv")
conditions = pd.read_csv(DATA_ROOT/"conditions"/"conditions.csv")

def to_age_years(b):
    try:
        d = pd.to_datetime(b).date(); t = date.today()
        return t.year - d.year - ((t.month, t.day) < (d.month, d.day))
    except Exception: return None

pat = (patients.rename(columns={"Id":"patient_id","GENDER":"gender","BIRTHDATE":"birthdate"})
               [["patient_id","gender","birthdate"]])
pat["age_years"] = pat["birthdate"].map(to_age_years)

conditions["active"] = conditions.get("STOP", pd.Series([None]*len(conditions))).isna()
cond = (conditions.rename(columns={"PATIENT":"patient_id","DESCRIPTION":"description",
                                  "CODE":"code","START":"start","STOP":"stop"})
                 [["patient_id","description","code","start","stop","active"]])

condL = (cond[cond["active"]]
         .groupby("patient_id")
         .apply(lambda d: d.head(5).to_dict(orient="records"))
         .rename("conditions").reset_index())

df = pat.merge(condL, on="patient_id", how="left")
df["conditions"] = df["conditions"].apply(lambda x: x if isinstance(x, list) else [])

def make_payload_json(r):
    return json.dumps({
        "patient_id": r["patient_id"],
        "demographics": {
            "gender": r["gender"],
            "birthdate": r["birthdate"],
            "age_years": int(r["age_years"]) if pd.notna(r["age_years"]) else None
        },
        "conditions": r["conditions"]
    })

df["payload_json"] = df.apply(make_payload_json, axis=1)

# optional labs_json from observations.csv
try:
    obs = pd.read_csv(DATA_ROOT/"observations"/"observations.csv")
    obs = obs.rename(columns={"PATIENT":"patient_id",
                              "DESCRIPTION":"test",
                              "VALUE":"value",
                              "UNITS":"unit"})
    labs = (obs.groupby("patient_id")
               .apply(lambda d: d.head(5)[["test","value","unit"]].to_dict(orient="records"))
               .rename("labs_json").reset_index())
    df = df.merge(labs, on="patient_id", how="left")
    df["labs_json"] = df["labs_json"].apply(lambda x: json.dumps(x) if isinstance(x, list) else "")
except FileNotFoundError:
    df["labs_json"] = ""

out = df[["patient_id","payload_json","labs_json"]].head(100)
out.to_csv(OUT, index=False, encoding="utf-8")
print("Wrote:", OUT.resolve())
