import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq

# ========== CONFIGURATION ==========
st.set_page_config(page_title="Lead Scoring AI", page_icon="📊", layout="wide")

# ========== API KEYS ==========
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Initialize Groq client
client = Groq(api_key="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")

# ========== Load ML Model ==========
@st.cache_resource
def load_model():
    return joblib.load("lead_scoring_model.pkl")

# ========== Data Cleaning ==========
def clean_revenue(value):
    if pd.isna(value): return np.nan
    value = str(value).replace('$', '').replace(',', '').strip().upper()
    if 'M' in value:
        return float(value.replace('M', '')) * 1e6
    elif 'B' in value:
        return float(value.replace('B', '')) * 1e9
    try:
        return float(value)
    except:
        return np.nan

def clean_employee_count(value):
    if pd.isna(value): return np.nan
    value = str(value).replace('+', '').replace(',', '').strip()
    try:
        return float(value)
    except:
        return np.nan

# ========== LLM Prompt ==========
def get_llm_explanation_groq(prompt_text):
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a top business analyst. Explain clearly why this lead received its score based on features."},
                {"role": "user", "content": f"Explain the lead score for the following business features:\n{prompt_text}"}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Groq API error: {e}")

# ========== Build Prompt ==========
def build_feature_summary(row, top_features):
    return "\n".join([f"{f}: {row.get(f, 'unknown')}" for f in top_features])

# ========== MAIN APP ==========
def main():
    st.markdown("<h1 style='text-align: center;'>🚀 AI-Powered Lead Scoring Tool</h1>", unsafe_allow_html=True)
    st.markdown("       Upload your business lead data and understand what drives their score using powerful LLM explanations.")

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/Graph_icon.svg", width=120)
        st.markdown("## 📁 Upload")
        uploaded_file = st.file_uploader("Upload Leads CSV", type=["csv"])
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit + Groq")

    if not uploaded_file:
        st.info("📌 Upload a CSV file from the sidebar to begin.", icon="📂")
        return

    features = [
        'Industry', 'Product/service', 'Business type',
        'Employee count', 'Revenue', 'Year founded',
        'BBB rating', "Owner's title", 'Source'
    ]

    df = pd.read_csv(uploaded_file)
    missing = [col for col in features if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        return

    df['Revenue'] = df['Revenue'].apply(clean_revenue)
    df['Employee count'] = df['Employee count'].apply(clean_employee_count)
    df['Year founded'] = pd.to_numeric(df['Year founded'], errors='coerce')
    X_input = df[features]

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    model = load_model()

    try:
        df["Lead Score"] = model.predict(X_input)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return

    st.success("✅ Lead scores generated!")
    st.bar_chart(df["Lead Score"])

    # --- Top Feature Importance ---
    st.subheader("📌 Top Features Used by Model")
    importances = model.named_steps["regressor"].feature_importances_
    top_features = [f for f, _ in sorted(zip(features, importances), key=lambda x: -x[1])[:5]]

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 🔎 Explain a Lead")
        index = st.number_input("Pick lead index", 0, len(df)-1, step=1)

        if st.button("🧠 Generate Explanation"):
            row = df.iloc[index]
            summary = build_feature_summary(row, top_features)

            with st.expander("🧬 Feature Summary"):
                st.code(summary)

            with st.spinner("💬 Thinking..."):
                try:
                    explanation = get_llm_explanation_groq(summary)
                    st.success("✅ Explanation Generated")
                    st.markdown("### 📝 Explanation")
                    st.markdown(explanation)
                except Exception as e:
                    st.error(f"Groq error: {e}")

    with col2:
        st.markdown("### 📊 Lead Summary")
        st.dataframe(df.iloc[[index]][features + ["Lead Score"]], use_container_width=True)

if __name__ == "__main__":
    main()
