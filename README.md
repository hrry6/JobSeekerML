# JobSeekerML — Deploy Guide (Streamlit)

**Overview:** This repository contains a job-recommendation service. Use `streamlit_app.py` as the Streamlit entrypoint to serve an interactive UI. This README explains how to run locally and deploy to Streamlit Community Cloud.

**Prerequisites:**
- Python 3.8+ and `pip`.
- A GitHub repository with this project (Streamlit Cloud reads from GitHub).

**Files of interest:**
- `streamlit_app.py`: Streamlit app entrypoint (main file for Streamlit Cloud).
- `requirements.txt`: Python dependencies. Ensure `streamlit` is listed.
- `model/model_rekomendasi.pkl`: trained model (commit to repo or host externally).
- `data/jobs.csv`: job dataset (optional; can set `LOCAL_JOBS_CSV` to point to CSV URL).

## Run locally
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. If the model is not committed, set `MODEL_URL` to a direct download URL (S3, GCS, raw GitHub):
```bash
export MODEL_URL="https://example.com/model_rekomendasi.pkl"
```
If CSV is not committed, set `LOCAL_JOBS_CSV` likewise.
3. Start Streamlit:
```bash
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud
1. Commit & push your repo to GitHub:
```bash
git add .
git commit -m "Prepare for Streamlit deploy"
git push origin main
```
2. On https://share.streamlit.io create a new app and select your GitHub repo and branch.
3. Set the **Main file** to `streamlit_app.py` and configure Environment variables (App Settings → Advanced):
   - `MODEL_URL` — (optional) direct URL to `model_rekomendasi.pkl` if not committing the model.
   - `LOCAL_JOBS_CSV` — (optional) direct URL to a CSV if not committing `data/jobs.csv`.
4. Deploy and open the app. If deployment fails, check Logs in the Streamlit Cloud dashboard.

## Notes & Troubleshooting
- Do NOT set the Streamlit main file to `app.py` if it contains `Flask` usage and `app.run(...)`. Streamlit runs `streamlit run <file>` and a Flask `app.run()` will conflict and be terminated (SIGTERM).
- Large model files: prefer storing on S3/GCS/Hugging Face and set `MODEL_URL`. You can also use Git LFS, but Streamlit Cloud may have repo size limits.
- Missing dependency errors: add the package to `requirements.txt`, commit, and redeploy.
- If the app times out while downloading large files, pre-download and commit small test files first to verify the app runs.

## Optional: Docker (not required for Streamlit Cloud)
If you want to run via Docker (for other hosts) ensure Dockerfile runs `streamlit run streamlit_app.py` and exposes port `8501`.

---
If you want, I can also: (A) create a small example `data/jobs.csv` for testing, or (B) create a short `README` section showing how to host the model on S3 and use `MODEL_URL`.
