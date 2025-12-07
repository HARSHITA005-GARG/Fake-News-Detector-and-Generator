# 📰 Fake News Generator & Detector  
### GPT-2 Fake News Generator + DistilBERT Fake News Classifier  
### FastAPI Backend + Streamlit Frontend + Full Deployment Guide

---

## 📌 Overview

This project is a complete **end-to-end NLP + MLOps system** that can:

- **Generate fake news headlines** using GPT-2  
- **Detect fake vs real headlines** using DistilBERT  
- Provide a **FastAPI backend** for serving ML models  
- Provide a **Streamlit UI** for user-friendly interaction  
- Be easily deployed on **Render** (backend) and **Streamlit Cloud** (frontend)

This repository is suitable for **academic submissions, ML portfolios, and real-world demonstrations**.

---

## 🚀 Features

### 🔷 Fake News Generator (GPT-2)
- Fine-tuned GPT-2 model for headline generation  
- Uses sampling (top-k, top-p, temperature)  
- Generates creative & realistic fake news text  

### 🔶 Fake News Detector (DistilBERT)
- Binary classifier (REAL = 0, FAKE = 1)  
- Achieves **99%+ accuracy, precision, recall, and F1-score**  
- Trained on FakeNewsNet-style dataset of headlines  

### 🔷 End-to-End Pipeline
Input prompt → Generate headlines → Detect fake/real → Return predictions + confidence

### 🔶 FastAPI Backend
Programmatic API with auto-generated Swagger docs.

Endpoints:

| Endpoint | Description |
|----------|-------------|
| `/generate` | Generate headlines via GPT-2 |
| `/detect` | Detect REAL or FAKE using DistilBERT |
| `/pipeline` | Generate + detect in one step |

### 🔷 Streamlit Frontend
User-friendly app that supports:

- Fake news generation  
- Fake news detection  
- Full pipeline testing  

---

## 📁 Project Structure
```
fake-news-project/
│
├── main.py # FastAPI backend
├── streamlit_app.py # Frontend UI
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Ignore unnecessary files
│
├── models/
│ ├── fakenews_model/ # DistilBERT classifier
│ └── gpt2_news/ # GPT-2 generator
│
└── render.yaml (optional for Render deployment)
```

---

## 🧠 Model Information

### 🟦 GPT-2 Generator
- Fine-tuned on news-style data  
- Produces headlines with realistic patterns  

### 🟧 DistilBERT Detector
Label mapping:
- `0` → REAL  
- `1` → FAKE  

Performance:
[](!images\image.png)


---

## ⚙️ Installation

### 1️. Clone the Repository

```bash
git clone https://github.com/<your-username>/fake-news-project.git
cd fake-news-project
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the backend

```
python -m uvicorn main:app --reload
# API docs available at:

http://127.0.0.1:8000/docs
```
### Run the Frontend (Streamlit)
```
streamlit run streamlit_app.py


Opens automatically at:

http://localhost:8501
```

## 🛠️ Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- FastAPI
- Streamlit
- Render
- Streamlit Cloud

## 🤝 Contributing

Pull requests are welcome!
Enhancements: UI improvements, dataset expansion, Docker support, etc.

## 📄 License

Open-source under the MIT License.
