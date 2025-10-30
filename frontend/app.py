"""
---------------------------------------------------
💬 STREAMLIT FRONTEND for JASP RAG

You type a query (e.g. “How to split data files in JASP?”)

You select a model (e.g. mistral:7b, llama3.2:3b, or phi3:mini) from the dropdown.

When you click “Generate Answer”, the frontend sends a POST request to your backend:
---------------------------------------------------
Run:
    poetry run streamlit run frontend/app.py
---------------------------------------------------
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/generate"

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
model = st.sidebar.selectbox("Select model:", ["mistral:7b", "llama3.2:3b", "phi3:mini"])
st.sidebar.markdown("---")

# ---------- Main UI ----------
st.title("🦁 RAG Assistant for JASP")
query = st.text_area("Ask a question about the JASP manual:", height=120, placeholder="e.g. How to split data files in JASP?")

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response... ⏳"):
            response = requests.post(API_URL, json={"query": query, "model": model})

        if response.status_code == 200:
            data = response.json()
            st.success("🍒 Answer generated successfully!")
            st.markdown("### ☄️ **Answer**")
            st.write(data["answer"])

            st.markdown("---")
            st.markdown("### 🥑 **Supporting Documents**")
            for i, doc in enumerate(data["documents"], 1):
                with st.expander(f"Document {i}: {doc['source']} (Page {doc['page']})"):
                    st.markdown(f"**Chapter:** {doc['chapter']}")
                    st.markdown(f"**Chunk ID:** {doc['chunk_id']}")
                    st.write(doc["text"])
        else:
            st.error(f"Error {response.status_code}: {response.text}")
