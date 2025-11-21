"""
---------------------------------------------------
💬 STREAMLIT FRONTEND for JASP RAG
---------------------------------------------------

Interactively query your local RAG system.

Main panel:
  • Type a question (e.g. “How to run repeated measures ANOVA in JASP”)
  • Click "Find docs" to retrieve relevant chunks (PDF pages, videos, GitHub files)
  • A big window shows the preview of the selected *original* document page

Sidebar:
  • Select an Ollama model (e.g. mistral:7b, llama3.2:3b, phi3:mini)
  • Click “Generate Answer” to get an AI-generated answer (optional, can be slow)

Run:
    poetry run streamlit run frontend/app.py
---------------------------------------------------
"""

import streamlit as st
import requests

# --------- Backend endpoints (adapt if needed) ----------
RETRIEVE_URL = "http://127.0.0.1:8000/retrieve"
GENERATE_URL = "http://127.0.0.1:8000/generate"  # or /generate_using_docs if you have that
PDF_PREVIEW_BASE = "http://127.0.0.1:8000/preview/pdf"  # /{pdf_id}/{page}.png
PDF_FULL_BASE = "http://127.0.0.1:8000/docs"            # /{pdf_id}.pdf


# -------------------------------------------------------
# Helper: PDF viewer – show original PDF page as image
# -------------------------------------------------------

def pdf_viewer(doc: dict):
    """
    Render the *original* PDF using a public source_url from metadata.
    Uses Option 1: iframe embedding.

    Expects doc to contain:
      - source_url (full PDF URL)
      - page or page_number (optional)
    """
    pdf_url = doc.get("source_url")
    if not pdf_url:
        st.warning("No 'source_url' found in metadata — cannot show PDF preview.")
        return

    # Page jump (works in most browsers / PDF viewers)
    page = int(doc.get("page") or doc.get("page_number") or 1)
    pdf_url_with_page = f"{pdf_url}#page={page}"

    st.markdown(f"### 📄 PDF Preview — Page {page}")

    # ----- Show full PDF using iframe -----
    st.markdown(
        f"""
        <iframe
            src="{pdf_url_with_page}"
            width="100%"
            height="900px"
            style="border: none;"
        ></iframe>
        """,
        unsafe_allow_html=True
    )

    # Optional: link to open PDF in a new tab
    st.markdown(
        f"[🔗 Open full PDF in new tab]({pdf_url})",
        unsafe_allow_html=False
    )


# -------------------------------------------------------
# Helper: Video viewer
# -------------------------------------------------------
def video_viewer(doc: dict):
    """
    Display YouTube video with timestamp jumping using iframe.
    """

    title = doc.get("title", doc.get("source", "Video"))
    st.markdown(f"#### 🎥 {title}")

    video_link = doc.get("video_link")
    if not video_link:
        st.info("No video link available in metadata.")
        return

    second_offset = doc.get("second_offset") or 0
    timestamp = doc.get("timestamp")

    # Convert typical YouTube link to embed format
    # from: https://www.youtube.com/watch?v=XXXX
    # to:   https://www.youtube.com/embed/XXXX
    embed_url = video_link.replace("watch?v=", "embed/")

    # Add time offset using YouTube embed format: ?start=NNN
    iframe_url = f"{embed_url}?start={second_offset}"

    st.markdown(
        f"""
        <iframe 
            width="100%" 
            height="450"
            src="{iframe_url}" 
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
        """,
        unsafe_allow_html=True,
    )


    if video_link:
        st.markdown(
            f'<a href="{video_link}" target="_blank">🔗 Open on YouTube</a>',
            unsafe_allow_html=True
        )


# -------------------------------------------------------
# Helper: Markdown / GitHub viewer
# -------------------------------------------------------
def markdown_viewer(doc: dict):
    """
    Display markdown or GitHub file content.
    Assumes doc has: content (markdown) and optional repo_url.
    """
    title = doc.get("title") or doc.get("source", "Markdown file")
    st.markdown(f"#### 📘 {title}")

    repo_url = doc.get("repo_url")
    if repo_url:
        st.markdown(f"[🔗 Open in GitHub]({repo_url})")

    content = doc.get("content") or doc.get("text") or ""
    if content:
        st.markdown("---")
        st.markdown(content)
    else:
        st.info("No content available in metadata.")





# -------------------------------------------------------
# Helper: Render a single retrieved document in the big window
# -------------------------------------------------------
def render_doc_preview(doc: dict):
    source_type = doc.get("source_type", "pdf")  # default to pdf if missing

    if source_type == "pdf":
        pdf_viewer(doc)
    elif source_type in ("video", "video_transcript"):
        video_viewer(doc)
    elif source_type in ("markdown", "code", "github"):
        markdown_viewer(doc)
    else:
        # Fallback: just show text
        st.markdown("#### 📄 Document Preview")
        st.write(doc.get("content") or doc.get("text") or "(no preview available)")


# -------------------------------------------------------
# Main App
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="JASP RAG – Retrieval-first UI", layout="wide")

    # ---------- Sidebar ----------
    st.sidebar.title("🦞 Settings")

    model = st.sidebar.selectbox(
        "Select model (Ollama):",
        ["mistral:7b", "llama3.2:3b", "phi3:mini"],
    )

    st.sidebar.markdown("---")
    generate_answer_btn = st.sidebar.button("✨ Generate Answer")


        # ---------- AI Answer block under the Generate Answer button ----------
    st.sidebar.markdown("---")

    if st.session_state["answer"]:
        st.sidebar.markdown("### 🤖 AI Generated Answer (bonus)")
        st.sidebar.markdown(st.session_state["answer"])

        if st.session_state["logs"]:
            with st.sidebar.expander("🪵 Logs / Debug Info"):
                for line in st.session_state["logs"]:
                    st.sidebar.text(line)


    # ---------- Main Layout ----------
    st.title("🦁 JASP RAG Protocol – Document Search")

    # Query input
    query = st.text_area(
        "Ask a question:",
        height=120,
        placeholder="e.g. How to run repeated measures ANOVA in JASP?",
        key="query_input",
    )

    # Buttons row
    find_docs_col, _ = st.columns([1, 3])
    with find_docs_col:
        find_docs = st.button("🔍 Find docs")

    # ---------- Session state ----------
    if "retrieved_docs" not in st.session_state:
        st.session_state["retrieved_docs"] = []
    if "selected_doc_idx" not in st.session_state:
        st.session_state["selected_doc_idx"] = 0
    if "answer" not in st.session_state:
        st.session_state["answer"] = None
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    # ---------- Handle "Find docs" (retrieval only) ----------
    if find_docs:
        if not query.strip():
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Retrieving relevant documents..."):
                try:
                    resp = requests.post(RETRIEVE_URL, json={"query": query})
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")
                else:
                    if resp.status_code != 200:
                        st.error(f"Backend error {resp.status_code}: {resp.text}")
                    else:
                        data = resp.json()
                        # Adapt keys to your backend shape: "results" vs "sources"
                        docs = data.get("results") or data.get("sources") or []
                        st.session_state["retrieved_docs"] = docs
                        st.session_state["selected_doc_idx"] = 0  # reset selection



    # ---------- Layout for docs list + big preview ----------
    docs = st.session_state["retrieved_docs"]

    if docs:

       # =====================================================
        # 1) Retrieved Documents (full width ABOVE preview)
        # =====================================================
        st.subheader("📚 Retrieved Documents")

        options = []
        for i, doc in enumerate(docs):
            rank = doc.get("rank", i + 1)
            stype = doc.get("source_type", "pdf")

            # --- Raw doc name (PDF filename, video ID, GitHub file, etc.) ---
            raw_name = (
                doc.get("pdf_id")
                or doc.get("title")
                or doc.get("markdown_file")
                or doc.get("github_name")
                or "unknown"
            )

            # Keep raw_name short
            if isinstance(raw_name, str) and len(raw_name) > 40:
                raw_name = raw_name[:40] + "..."

            # --- Section name / title ---
            section_name = (
                doc.get("timestamp")
                or doc.get("title")
                or "Untitled section"
            )

            # Build label like:
            # Rank 3 | video | APRaBFC2lEQ | Introduction to JASP
            label = f"Rank {rank} | {stype} | {raw_name} | {section_name}"

            options.append(label)

        selected_label = st.radio(
            "Select a document to preview:",
            options,
            index=st.session_state["selected_doc_idx"],
        )
        st.session_state["selected_doc_idx"] = options.index(selected_label)

        st.markdown("---")

       
        # =====================================================
        # 2) Document Preview (full width BELOW doc list)
        # =====================================================
        st.subheader("🔍 Document Preview")

        selected_doc = docs[st.session_state["selected_doc_idx"]]
        
        #st.write("DEBUG retrieved doc:", selected_doc)

        render_doc_preview(selected_doc)


    else:
        st.info("No documents yet. Enter a query and click **Find docs** to start.")

    # ---------- Handle "Generate Answer" in sidebar ----------
    if generate_answer_btn:
        if not query.strip():
            st.sidebar.warning("Please enter a query in the main panel first.")
        else:
            with st.spinner("Generating AI answer (this may take a while)..."):
                try:
                    payload = {
                        "query": query,
                        "model": model,
                        # If your backend supports using provided docs, you can send them:
                        # "retrieved_docs": st.session_state["retrieved_docs"],
                    }
                    resp = requests.post(GENERATE_URL, json=payload)
                except Exception as e:
                    st.sidebar.error(f"Error contacting backend: {e}")
                else:
                    if resp.status_code != 200:
                        st.sidebar.error(
                            f"Backend error {resp.status_code}: {resp.text}"
                        )
                    else:
                        data = resp.json()
                        st.session_state["answer"] = data.get("answer")
                        st.session_state["logs"] = data.get("logs", [])




if __name__ == "__main__":
    main()
