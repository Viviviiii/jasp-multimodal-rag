# 📘 JASP Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) system for the **JASP manual**.  
Supports **text + image ingestion** (via BLIP + CLIP), storage in **Chroma**, and querying through **FastAPI** and **Streamlit**.

---

## ⚙️ Setup

### Prerequisites
- Python **3.11** (3.12 works, but 3.13 is not yet recommended)
- [Poetry 2.x](https://python-poetry.org/docs/#installation)
- [Ollama](https://ollama.com) (for local LLMs like `llama3` or `mistral:7b-instruct`)

---

## 🚀 Usage

Install dependencies

        poetry lock
        poetry install

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


###### update: When you make changes in VS Code:
git add .
git commit -m "full pipeline version-5 "
git push


|-- README.md
|-- data
|-- src
    |-- ingestion
    |-- splitting
    |-- pipelines
    |-- retrieval
    |-- generation
|-- backend_api
|   `-- main.py
|-- frontend
|   `-- app.py
|-- notebooks
|-- poetry.lock
|-- pyproject.toml
|-- .gitignore