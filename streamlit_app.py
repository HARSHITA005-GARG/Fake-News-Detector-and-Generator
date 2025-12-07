import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"  # change after deployment

st.set_page_config(page_title="Fake News Generator & Detector", layout="centered")

st.title("📰 Fake News Generator & Detector")
st.write("Generate fake news and detect whether a headline is real or fake.")

# ---------------------------
# Fake News Generation UI
# ---------------------------
st.header("Generate Fake News")

prompt = st.text_input("Enter a prompt:", "Breaking News:")
num = st.slider("Number of headlines to generate:", 1, 5, 3)

if st.button("Generate"):
    with st.spinner("Generating fake news..."):
        response = requests.post(
            f"{FASTAPI_URL}/generate",
            json={"prompt": prompt, "num": num}
        )
        data = response.json()
        st.subheader("Generated Headlines:")
        for idx, item in enumerate(data["generated_news"], 1):
            st.write(f"**{idx}. {item}**")

# ---------------------------
# Fake News Detector UI
# ---------------------------
st.header("Fake News Detection")

text_to_check = st.text_area("Enter a headline to classify:")

if st.button("Detect"):
    with st.spinner("Analyzing headline..."):
        response = requests.post(
            f"{FASTAPI_URL}/detect",
            json={"text": text_to_check}
        )
        data = response.json()
        st.subheader("Prediction:")
        st.write(f"📝 **Text:** {data['text']}")
        st.write(f"🔍 **Prediction:** `{data['prediction']}`")
        st.write(f"📊 **Confidence:** {round(data['confidence'], 3)}")

# ---------------------------
# Full Pipeline UI
# ---------------------------
st.header("🔷 Full Pipeline (Generate + Detect)")

pipe_prompt = st.text_input("Pipeline Prompt:", "Breaking News:")
pipe_num = st.slider("Generate how many headlines?", 1, 5, 3)

if st.button("Run Pipeline"):
    with st.spinner("Running pipeline..."):
        response = requests.post(
            f"{FASTAPI_URL}/pipeline",
            json={"prompt": pipe_prompt, "num": pipe_num}
        )
        data = response.json()
        st.subheader("Pipeline Output:")
        for result in data["results"]:
            st.write(f"**📰 {result['headline']}**")
            st.write(f"- Prediction: `{result['prediction']}`")
            st.write(f"- Confidence: {round(result['confidence'], 3)}")
            st.write("---")
