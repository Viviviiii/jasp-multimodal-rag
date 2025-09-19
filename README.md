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

### 1. Install dependencies
```bash
poetry lock
poetry install

### 2. Activate environment
```bash
poetry env activate


### 3. Run backend (FastAPI)
```bash
poetry run uvicorn main:app --reload --port 8000

### 4. Run frontend (Streamlit)
```bash
poetry run streamlit run app.py

### 5.You can also query directly:
```bash
poetry run python src/ingestion/data_create.py
poetry run python src/pipelines/data_query.py "How do I run multiple regression in JASP?" --model mistral:7b-instruct --topk 1

###### update: When you make changes in VS Code:
git add .
git commit -m "Describe your change"
git push
