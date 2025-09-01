# clinical-ops-flan-t5-agent
_streamlit, llm, flan-t5, transformers, pydantic, clinical-trials, healthcare, agent, genai, python_

# 🩺 Clinical Trial Ops Assistant (Agentic, FLAN-T5)

A compact demo that turns clean clinical tables into **structured JSON** and automates:

- **Patient narratives** (4–6 sentence, patient-friendly notes)
- **Eligibility checks** (simple inclusion/exclusion rules)
- **Lab flagging** (out-of-range alerts + quick chart)
- **Batch mode** (CSV in → results CSV out)
- **DOCX export** (download narratives)
- **RAG stubs** (placeholders to add protocol/SOP retrieval next)

> **No model training required.** Uses a pre-trained, instruction-tuned LLM (**FLAN-T5**) for controlled generation.

---

## 🧱 Project layout
├─ streamlit_app.py # Streamlit UI with Narrative / Eligibility / Labs / Batch tabs
├─ app.py # (optional) Gradio version of the assistant
├─ requirements.txt # Cloud-friendly deps (pins avoid build issues)
├─ clinical_batch_examples.csv
├─ notebooks/ # (optional) EDA & data-cleaning notes
└─ data/ # (optional) local pointers to raw CSVs (patients, conditions, etc.)

---

## 🧼 Data cleaning (what this repo demonstrates)

**Goal:** standardize raw clinical CSVs (e.g., Synthea-like) into tidy, query-ready JSON used by the LLM tools.

**Typical input tree**
root/
├─ patients/ *.csv
├─ conditions/ *.csv
├─ encounters/ *.csv
├─ medications/ *.csv
├─ observations/ *.csv
├─ allergies/ *.csv
├─ careplans/ *.csv
├─ immunizations/ *.csv
└─ procedures/ *.csv

**Key rules**
- **Dates** → enforce `YYYY-MM-DD`. Invalid → drop or set safe default (`1900-01-01`).
- **Types** → cast numeric/bool; keep medical codes as strings.
- **Duplicates** → drop exact dups; when relevant, keep latest by `(id, date)`.
- **Outputs** → write **Bronze** (raw→parquet) and **Silver** (cleaned) folders for downstream apps.

**Example (Spark)**

**from pyspark.sql import functions as F

DATE_RX = r"\d{4}-\d{2}-\d{2}"

patients_clean = (patients
    .withColumn(
        "BIRTHDATE",
        F.when(F.col("BIRTHDATE").rlike(DATE_RX), F.col("BIRTHDATE"))
         .otherwise(F.lit("1900-01-01"))
    )
    .dropDuplicates()
)

conditions_clean = (conditions
    .filter(F.col("START").rlike(DATE_RX))
    .dropDuplicates()
)**

**LLM used**

Model: google/flan-t5-base (optionally …-large on a strong local GPU)

Why: small, stable, good instruction following → reliable short summaries

Typical limits: ~512 tokens input context; generate ≤128–256 tokens

Deterministic decoding (default): num_beams=4, do_sample=False, length_penalty=0.9, no_repeat_ngram_size=3

Safety line: patient-facing messages always end with
“Contact your care team if symptoms change.”

# 1) create env
conda create -n clinicalops python=3.11 -y
conda activate clinicalops

# 2) install deps
pip install -r requirements.txt

# 3a) run Streamlit UI (recommended)
streamlit run streamlit_app.py

# 3b) or run the Gradio app
python app.py
How to use the app
1) Narrative (single patient JSON)
{
  "patient_id": "X001",
  "demographics": { "age_years": 55, "gender": "M" },
  "conditions": [{ "description": "Diabetes", "active": true }]
}
**Under the hood (agentic view)**

Tool 1 — Narrative: builds a prompt from JSON and queries FLAN-T5.

Tool 2 — Eligibility: pure Python rules (age, includes, excludes).

Tool 3 — Lab flagger: simple range checks + chart.

This is “agentic” because the UI orchestrates the right tool per task and enforces prompt discipline (no free-form model chatter).
