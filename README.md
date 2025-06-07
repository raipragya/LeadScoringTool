# 🧠 Lead Scoring & Explainability Tool for SaaSquatch Leads

## 🚀 Overview

This project enhances the lead generation pipeline by scoring enriched leads (from [SaaSquatchLeads.com](https://www.saasquatchleads.com/)) using a trained machine learning model, and providing **human-readable explanations** for non-technical users. It prioritizes **high-quality leads** to help sales teams take faster and smarter action.

> ✅ Built in 5 hours for the Caprae Capitals Leadgen Challenge  
> ✅ No scraping needed — this tool works on *already-enriched* SaaSquatch data  
> ✅ Empowers sales teams with plain-English insights

---

## 🧰 Features

- 📥 Upload SaaSquatch CSV files (enriched lead data)
- 🎯 Assign a **lead score** using a trained `RandomForestRegressor`
- 🔍 Get **plain-language explanations** for why a lead is good or bad
- 📊 Filter and sort leads interactively

---
## Public URL:
https://leadscoringtool-o5fat4gze5tryzhyehufu3.streamlit.app/

## 📦 Installation & Setup

```bash
git clone https://github.com/your-username/lead-scoring-tool.git
cd lead-scoring-tool
pip install -r requirements.txt
python train_model.py(make sure your model is in the same directory)
get your GROQ_API_KEY from groq cloud
streamlit run app.py
