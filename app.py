# app.py
import streamlit as st
from src.pipelines.data_query import run_query, build_rag_chain

st.set_page_config(page_title="JASP Manual Assistant", page_icon="📘")

st.title("📘 JASP Manual Q&A")
st.write("Ask a question about the JASP manual (RAG-powered).")

query_text = st.text_input("Enter your question:")
model = st.selectbox("Choose a model:", ["llama3", "mistral:7b-instruct"])
topk = st.slider("Chunks to retrieve:", 1, 6, 3)

if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            chain = build_rag_chain(model, topk)
            result = chain.invoke({"query": query_text})

            st.subheader("Answer")
            st.write(result["result"])

            if result.get("source_documents"):
                st.subheader("Sources")
                for doc in result["source_documents"]:
                    page = doc.metadata.get("page")
                    src = f"Page {page}" if page else doc.metadata.get("path", "Unknown source")
                    st.write(f"- {src}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
