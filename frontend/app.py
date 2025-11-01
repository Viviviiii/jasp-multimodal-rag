"""
---------------------------------------------------
💬 STREAMLIT FRONTEND for JASP RAG
---------------------------------------------------

Interactively query your local RAG system.

You can:
  • Type a question (e.g. “How to split data files in JASP?”)
  • Select an Ollama model (e.g. mistral:7b, llama3.2:3b, phi3:mini)
  • Click “Generate Answer” to retrieve and generate the result.

Run:
    poetry run streamlit run frontend/app.py
---------------------------------------------------
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/generate"

# ---------- Sidebar ----------
st.sidebar.title("🦞 Settings")
model = st.sidebar.selectbox("Select model:", ["mistral:7b", "llama3.2:3b", "phi3:mini"])
st.sidebar.markdown("---")

# ---------- Main UI ----------
st.title("🦁 JASP RAG Assistant")
query = st.text_area(
    "Ask a question about the JASP manual:",
    height=120,
    placeholder="e.g. How to split data files in JASP?"
)

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response... 🐦‍🔥"):
            response = requests.post(API_URL, json={"query": query, "model": model})

        if response.status_code == 200:
            data = response.json()

            st.success("🔥 Answer generated successfully!")
            st.markdown("### ☄️ **Answer**")
            st.markdown(data["answer"])

            st.markdown("---")
            st.markdown("### 🦄 **Source Documents**")

            for src in data.get("sources", []):
                with st.expander(f"Rank {src['rank']}: {src['source']} (Page {src['page']})"):
                    st.markdown(f"**Chunk ID:** {src['chunk_id']}")
                    st.markdown(f"**Score:** {round(src['score'], 3) if src['score'] else 'N/A'}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
