import os
import re
import joblib
import requests
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def custom_tokenizer(text):
    if not text:
        return []
    text = re.sub(r'[\,\;\/]', ' ', text.lower())
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b|\b[a-zA-Z]\+\+|\b[a-zA-Z]#', text)
    return [t.strip() for t in tokens if t.strip() and t.strip() != '.']


@st.cache_resource
def load_model():
    model_path = os.path.join('model', 'model_rekomendasi.pkl')
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f'Gagal memuat model lokal: {e}')
            st.stop()

    model_url = os.environ.get('MODEL_URL')
    if model_url:
        try:
            os.makedirs('model', exist_ok=True)
            with requests.get(model_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return joblib.load(model_path)
        except Exception as e:
            st.error(f'Gagal mengunduh model dari MODEL_URL: {e}')
            st.stop()

    st.error('Model tidak ditemukan. Commit `model/model_rekomendasi.pkl` atau set env `MODEL_URL`.')
    st.stop()


@st.cache_data
def load_dataset():
    candidates = []
    env_csv = os.environ.get('LOCAL_JOBS_CSV')
    if env_csv:
        candidates.append(env_csv)
    candidates.append(os.path.join('data', 'jobs.csv'))

    for root, _, files in os.walk('.'):
        for f in files:
            if f.lower().endswith('.csv'):
                p = os.path.join(root, f)
                if p not in candidates:
                    candidates.append(p)

    required = ['jobtitle', 'company', 'joblocation_address', 'skills']
    for file_csv in candidates:
        if os.path.exists(file_csv):
            try:
                df = pd.read_csv(file_csv)
                if not all(col in df.columns for col in required):
                    continue
                df = df[required].dropna()
                df['skills_lower'] = df['skills'].str.lower()
                return df
            except Exception:
                continue

    st.error('Tidak menemukan CSV jobs yang valid. Letakkan data di `data/jobs.csv` atau set `LOCAL_JOBS_CSV`.')
    st.stop()


def get_recommendations(user_skills, model_ai, df_kaggle):
    user_skills_clean = user_skills.lower()
    prediksi_label = model_ai.predict([user_skills_clean])[0]
    probabilitas = model_ai.predict_proba([user_skills_clean])[0]
    skor_keyakinan_ai = float(probabilitas[prediksi_label])

    df_sampel = df_kaggle.copy()
    tfidf_vectorizer = model_ai.named_steps['tfidf']
    matrix_jobs = tfidf_vectorizer.transform(df_sampel['skills_lower'])
    matrix_user = tfidf_vectorizer.transform([user_skills_clean])
    skor_kemiripan_teks = cosine_similarity(matrix_user, matrix_jobs).flatten()

    bonus_prediksi = []
    kata_kunci_it = ['developer', 'java', 'python', 'react', 'sql', 'data', 'web', 'php']
    for _, row in df_sampel.iterrows():
        is_it_job = any(kata in str(row['skills_lower']) for kata in kata_kunci_it)
        if (prediksi_label == 0 and is_it_job) or (prediksi_label == 1 and not is_it_job):
            bonus_prediksi.append(0.2)
        else:
            bonus_prediksi.append(0.0)

    df_sampel['Final_Score'] = (skor_kemiripan_teks * 0.5) + \
                               (skor_keyakinan_ai * 0.3) + \
                               (pd.Series(bonus_prediksi, index=df_sampel.index) * 0.2)

    rekomendasi_teratas = df_sampel.sort_values(by='Final_Score', ascending=False).head(5)

    results = []
    for _, row in rekomendasi_teratas.iterrows():
        persentase_match = max(45.0, min(99.0, float(row['Final_Score']) * 100))
        raw_skills = str(row['skills'])
        tokens = custom_tokenizer(raw_skills)
        unique_skills = list(dict.fromkeys(tokens))
        clean_skills = ", ".join(unique_skills[:6]).upper()
        skills_display = clean_skills if clean_skills else raw_skills
        results.append({
            'Job Title': row['jobtitle'],
            'Company': row['company'],
            'Location': row['joblocation_address'],
            'Skills': skills_display,
            'Match Score': f"{persentase_match:.1f}%"
        })

    return pd.DataFrame(results)


def main():
    st.title('JobSeekerML — Rekomendasi Lowongan Kerja')
    st.write('Masukkan keahlian Anda (contoh: python, data analysis, sql)')
    user_input = st.text_area('Keahlian Anda', height=120)

    with st.spinner('Memuat model dan dataset...'):
        model_ai = load_model()
        df_kaggle = load_dataset()

    if st.button('Cari Rekomendasi'):
        if not user_input or not user_input.strip():
            st.warning('Masukkan keahlian terlebih dahulu.')
        else:
            with st.spinner('Menghitung rekomendasi...'):
                df_result = get_recommendations(user_input, model_ai, df_kaggle)
                if df_result.empty:
                    st.info('Tidak ada rekomendasi ditemukan.')
                else:
                    st.table(df_result)


if __name__ == '__main__':
    main()
