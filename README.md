# clinical-ops-flan-t5-agent
streamlit, llm, flan-t5, transformers, pydantic, clinical-trials, healthcare, agent, genai, python
#  Clinical Trial Ops Assistant (Agentic, FLAN-T5)

A small, job-ready demo that turns raw clinical tables into compact patient JSON and automates:

- **Patient narratives** (4–6 sentence, patient-friendly notes)
- **Eligibility checks** (simple inclusion/exclusion rules)
- **Lab flagging** (out-of-range alerts + quick chart)
- **Batch mode** (CSV in → results CSV out)
- **DOCX export** (download narratives)
- **RAG stubs** (placeholders to add protocol/SOP retrieval next)

No model training required. Uses a pre-trained, instruction-tuned LLM (**FLAN-T5**) for controlled generation.

---

##  Live demo options

### A) Streamlit Community Cloud
1. Push this repo to GitHub (public).
2. Go to https://streamlit.io/cloud → **Deploy an app**.
3. Select your repo, set **Main file path** = `streamlit_app.py`.
4. Click **Deploy**.  
   _Note:_ Streamlit Cloud is CPU-only; default model is `google/flan-t5-base`.

### B) Run locally
```bash
# 1) create env (conda or venv)
pip install -r requirements.txt

# 2) start the app
streamlit run streamlit_app.py
