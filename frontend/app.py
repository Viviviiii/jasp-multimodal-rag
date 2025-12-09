
"""
---------------------------------------------------
7:
 Streamlit frontend for JASP RAG
---------------------------------------------------

This app is a thin UI layer on top of the FastAPI backend. It lets you:

Main panel:
  • Type a question (e.g. “How to run repeated measures ANOVA in JASP?”)
  • Click **"Search docs"** to retrieve relevant chunks
    (PDF pages, GitHub help files, video segments)
  • Browse a ranked list of results and view the selected source in a large
    preview area:
      – PDF: in-browser page view with page jump  
      – Video: YouTube player with timestamp  
      – GitHub: rendered Markdown from the original file

Sidebar:
  • Choose a retrieval mode (BM25 / vector / hybrid / fusion / fusion + rerank)
  • Select an Ollama model (e.g. `mistral:7b`, `llama3.2:3b`, `phi3:mini`) for generation task
  • Optionally click **"Generate Answer"** to run full RAG (retrieval + LLM) and
    see the answer directly in the sidebar, along with debug logs.

Backend endpoints used:
  • POST /retrieve
  • POST /generate
  • GET  /config/retrieval-modes

Run locally:

    poetry run streamlit run frontend/app.py

(Make sure the backend is running first: poetry run uvicorn backend_api.main:app --reload --port 8000 )

---------------------------------------------------
"""



import streamlit as st
import requests

# --------- Backend endpoints (adapt if needed) ----------
RETRIEVE_URL = "http://127.0.0.1:8000/retrieve"
GENERATE_URL = "http://127.0.0.1:8000/generate"
CONFIG_MODES_URL = "http://127.0.0.1:8000/config/retrieval-modes"

# Default modes in case /config/retrieval-modes is not available
DEFAULT_MODES = [
    "bm25",
    "vector",
   # "bm25_vector",
    "bm25_vector_fusion",
    "bm25_vector_fusion_rerank",
]
DEFAULT_MODE = "bm25_vector_fusion_rerank"


# -------------------------------------------------------
# Helper: PDF viewer – show original PDF page as iframe
# -------------------------------------------------------
def pdf_viewer(doc: dict):
    """
    Render the *original* PDF using a public source_url from metadata.
    Uses iframe embedding and jumps to the correct page.

    Expects doc to contain:
      - source_url (full PDF URL)
      - page or page_number (optional)
    """
    pdf_url = doc.get("source_url")
    if not pdf_url:
        st.warning("No 'source_url' found in metadata — cannot show PDF preview.")
        return

    raw_page = doc.get("page") or doc.get("page_number") or 1
    try:
        page = int(raw_page)
    except Exception:
        page = 1   # fallback if "?" or invalid

    pdf_url_with_page = f"{pdf_url}#page={page}"

    st.markdown(f"### Check details in Page {page}")
    st.markdown(
        f"[📖 Original PDF]({pdf_url})\n\n"
        "[📚 JASP Support Materials](https://jasp-stats.org/jasp-materials/)",
        unsafe_allow_html=False,
    )

    st.markdown(
        f"""
        <iframe
            src="{pdf_url_with_page}"
            width="100%"
            height="900px"
            style="border: none;"
        ></iframe>
        """,
        unsafe_allow_html=True,
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

    st.markdown(
        f"[📖 Original Video Link]({video_link})\n\n"
        "[📚 JASP Support Materials](https://jasp-stats.org/jasp-materials/)",
        unsafe_allow_html=False,
    )

    second_offset = doc.get("second_offset") or 0

    # Convert typical YouTube link to embed format
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


# -------------------------------------------------------
# Helper: Markdown / GitHub viewer
# -------------------------------------------------------
def to_raw_url(url: str):
    """
    Convert a GitHub webpage URL into raw.githubusercontent.com URL.
    """
    if "github.com" not in url:
        return url

    return (
        url.replace("github.com", "raw.githubusercontent.com")
           .replace("/blob/", "/")
    )


def markdown_viewer(doc: dict):
    title = doc.get("title") or doc.get("source", "Markdown file")
    github_url = doc.get("source_url") or doc.get("md_url")

    if not github_url:
        st.info("No markdown URL provided.")
        return

    st.markdown(f"#### Detail: {title}")
    st.markdown(
        f"[📖 Original GitHub file]({github_url})\n\n"
        "[📚 JASP Support Materials](https://jasp-stats.org/jasp-materials/)",
        unsafe_allow_html=False,
    )

    # Repair /tree/ GitHub URLs
    if "/tree/" in github_url:
        github_url = github_url.replace("/tree/", "/blob/")

    raw_url = to_raw_url(github_url)

    # Fetch markdown
    try:
        res = requests.get(raw_url)
        if res.status_code == 200:
            md_text = res.text
        else:
            st.error(f"HTTP {res.status_code} fetching: {raw_url}")
            return
    except Exception as e:
        st.error(f"Error loading markdown: {e}")
        return

    # Render markdown
    st.markdown("---")
    st.markdown(md_text)


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
# Sidebar: fetch and choose retrieval mode
# -------------------------------------------------------
def get_retrieval_mode_from_sidebar():
    """
    Fetch available retrieval modes from the backend (once) and
    render a selectbox at the top of the sidebar.

    Returns the selected mode as a string.
    """
    # Fetch config only once per session
    if "retrieval_modes" not in st.session_state:
        modes = DEFAULT_MODES
        default_mode = DEFAULT_MODE
        try:
            resp = requests.get(CONFIG_MODES_URL, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                modes = data.get("modes") or modes
                default_mode = data.get("default") or default_mode
        except Exception:
            # Fail silently – fall back to defaults
            pass

        st.session_state["retrieval_modes"] = modes
        st.session_state["retrieval_default_mode"] = default_mode

    modes = st.session_state["retrieval_modes"]
    default_mode = st.session_state["retrieval_default_mode"]

    # Compute default index safely
    try:
        default_index = modes.index(default_mode)
    except ValueError:
        default_index = 0 if modes else 0

    st.sidebar.markdown("### 🦞 Retrieval Mode Settings")
    mode = st.sidebar.selectbox(
        "Retrieval mode:",
        modes,
        index=default_index,
        help=(
            "Choose how the system retrieves relevant documents:\n\n"
            "• **bm25** – Classical keyword-based search.\n"
            "   - Fast and robust for exact terms.\n"
            "   - Works well when the query wording matches the document.\n\n"
            "• **vector** – Dense semantic search using embeddings.\n"
            "   - Captures meaning instead of exact words.\n"
            "   - Useful when the query is phrased differently than the text.\n\n"
            "• **bm25_vector_fusion** – Rank fusion of BM25 + vector search.\n"
            "   - Combines lexical and semantic strengths.\n"
            "   - More stable across different question types.\n\n"
            "• **bm25_vector_fusion_rerank** – Full retrieval pipeline.\n"
            "   - First fuses BM25 and vector rankings.\n"
            "   - Then reorders top results with a cross-encoder for maximum precision.\n"
            "   - Recommended as the most accurate mode.\n\n"
            "Tip: If unsure, choose the rerank mode for best overall quality."
        )

    )

    st.session_state["retrieval_mode"] = mode
    return mode


# -------------------------------------------------------
# Main App
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="JASP RAG – Retrieval-first UI", layout="wide")

    # ---------- Session state ----------
    if "retrieved_docs" not in st.session_state:
        st.session_state["retrieved_docs"] = []
    if "selected_doc_idx" not in st.session_state:
        st.session_state["selected_doc_idx"] = 0
    if "answer" not in st.session_state:
        st.session_state["answer"] = None
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if "retrieval_mode" not in st.session_state:
        st.session_state["retrieval_mode"] = DEFAULT_MODE

    # ---------- Sidebar ----------
    retrieval_mode = get_retrieval_mode_from_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.title("🦄 AI generated Answers")

    model = st.sidebar.selectbox(
        "Select model (Ollama):",
        ["mistral:7b", "llama3.2:3b", "phi3:mini"],
    )
    generate_answer_btn = st.sidebar.button("Generate Answer")

    # AI Answer block under the Generate Answer button
    if st.session_state["answer"]:
        st.sidebar.markdown("### Answer:")
        st.sidebar.markdown(st.session_state["answer"])

        if st.session_state["logs"]:
            with st.sidebar.expander("🦧 Logs / Debug Info"):
                for line in st.session_state["logs"]:
                    st.sidebar.text(line)

    # ---------- Main Layout ----------
    st.title("🦁 JASP RAG Protocol")
    st.caption(f"Current retrieval mode: `{retrieval_mode}`")

    # Query input
    query = st.text_area(
        "What do you want to know?",
        height=120,
        placeholder="e.g. How to run repeated measures ANOVA in JASP?",
        key="query_input",
    )

    # Buttons row
    find_docs_col, _ = st.columns([1, 3])
    with find_docs_col:
        find_docs = st.button("Search docs")

    status_box = st.empty()

    # ---------- Handle "Search docs" (retrieval only) ----------
    if find_docs:
        if not query.strip():
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("🦀 Searching in JASP Knowledge Base..."):
                try:
                    resp = requests.post(
                        RETRIEVE_URL,
                        json={"query": query, "mode": retrieval_mode},
                    )
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")
                else:
                    if resp.status_code != 200:
                        st.error(f"Backend error {resp.status_code}: {resp.text}")
                    else:
                        data = resp.json()
                        docs = data.get("results") or data.get("sources") or []
                        st.session_state["retrieved_docs"] = docs
                        st.session_state["selected_doc_idx"] = 0  # reset selection

    # ---------- Layout for docs list + big preview ----------
    docs = st.session_state["retrieved_docs"]

    if docs:
        # 1) Retrieved Documents (list)
        st.subheader("🦞 Docs may help:")

        options = []
        for i, doc in enumerate(docs):
            rank = doc.get("rank", i + 1)
            stype = doc.get("source_type", "pdf")

            raw_name = (
                doc.get("pdf_id")
                or doc.get("title")
                or doc.get("markdown_file")
                or doc.get("github_name")
                or "unknown"
            )

            if isinstance(raw_name, str) and len(raw_name) > 40:
                raw_name = raw_name[:40] + "..."

            section_name = (
                doc.get("start_page")
                or doc.get("page")
                or doc.get("section")
                or doc.get("timestamp")
                or doc.get("title")
                or "Untitled section"
            )

            label = f"Rank {rank} | {stype} | {raw_name} | {section_name}"
            options.append(label)

        selected_label = st.radio(
            "Select one to see details:",
            options,
            index=st.session_state["selected_doc_idx"],
        )
        st.session_state["selected_doc_idx"] = options.index(selected_label)

        # 2) Document Preview (full width below doc list)
        selected_doc = docs[st.session_state["selected_doc_idx"]]
        render_doc_preview(selected_doc)

    else:
        st.info("No docs found yet. Try a query and click 'Search docs'.")

    # ---------- Handle "Generate Answer" in sidebar ----------
    if generate_answer_btn:
        if not query.strip():
            status_box.warning("Please enter a query before generating an answer.")
        else:
            payload = {
                "query": query,
                "model": model,
                "mode": retrieval_mode,
            }

            status_box.info("🐝🐝🐝 AI is thinking...")

            try:
                resp = requests.post(GENERATE_URL, json=payload, timeout=60)
            except Exception as e:
                status_box.error(f"🐞 NETWORK ERROR: {e}")
                return

            if resp.status_code != 200:
                status_box.error(f"🐞 BACKEND ERROR {resp.status_code}: {resp.text}")
                return

            data = resp.json()
            st.session_state["answer"] = data.get("answer", "(No answer returned)")
            st.session_state["logs"] = data.get("logs", [])

            status_box.success("✔ Answer generated!")
            st.rerun()


if __name__ == "__main__":
    main()
