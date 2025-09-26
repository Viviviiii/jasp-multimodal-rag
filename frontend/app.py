"""
Streamlit frontend for the JASP RAG chatbot.

Features:
- Provides a simple web UI to query the ingested JASP manual.
- Sends requests to the FastAPI backend (`/ask` endpoint).
- Displays both the generated answer and the retrieved sources.
- Lets the user select which LLM model to use and how many chunks to retrieve (top-k).

Usage:
1. Start the FastAPI backend in a separate terminal:
       poetry run uvicorn backend.main:app --reload --port 8000

2. Start this Streamlit frontend:
       poetry run streamlit run frontend/app.py

3. Open the app in your browser:
       http://localhost:8501

4. Quit the app:
       Press CTRL+C in the terminal where the app is running.

Backend API docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc:      http://127.0.0.1:8000/redoc

Example:
- Type your question in the text area (e.g., "How do I run ANOVA in JASP?")
- Choose model (default: mistral:7b-instruct) and top-k value.
- Click "Ask" → the system queries the backend and shows an answer with sources.
"""



import streamlit as st
import requests

# Backend URL (adjust if running on another host/port)
API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="JASP RAG Chatbot", layout="wide")
st.title("📊 JASP Multimodal RAG Chatbot")

st.markdown(
    "Ask a question about the JASP manual. "
    "The system retrieves relevant manual text and image captions, "
    "then generates an answer."
)

# Sidebar settings
st.sidebar.header("Settings")
model = st.sidebar.selectbox(
    "LLM Model",
    options=["mistral:7b-instruct", "llama3"],
    index=0,
)
topk = st.sidebar.slider("Top-k chunks", min_value=1, max_value=5, value=2)

# User input
query = st.text_area("Enter your question:", placeholder="e.g., How do I run ANOVA in JASP?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying backend..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "model": model, "topk": topk},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Answer")
                    st.write(data.get("answer", ""))

                    st.subheader("Sources")
                    sources = data.get("sources", [])
                    if sources:
                        for s in sources:
                            st.markdown(f"- {s}")
                    else:
                        st.info("No sources found.")
                else:
                    st.error(f"Backend error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
