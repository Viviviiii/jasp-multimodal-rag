# JASP Multimodal RAG
A multimodal Retrieval-Augmented Generation (RAG) system for the JASP.  
Supports pdf, github,video, storage in Chroma, and querying through FastAPI** and Streamlit.


# Prerequisites
- Python 3.11 (3.12 works, but 3.13 is not yet recommended)
- [Poetry 2.x](https://python-poetry.org/docs/#installation)
- [Ollama](https://ollama.com) 
- llamaindex


# Usage

0. Install dependencies
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

Example:
- Type your question in the text area (e.g., "How do I run ANOVA in JASP?")
- Click "Find helpful Docs" → the system queries the backend and finds best match documents in the knowledge base.
- Choose model (default: mistral:7b).Click"Generate answer"
       → the system queries the backend and answer questions with reference.


# update: When you make changes in VS Code:
git add .
git commit -m "full pipeline version-5-metadata fixed "
git push


|-- README.md #High-level overview of the project, setup instructions, and how everything works.
|-- data      #Stores raw, processed, and intermediate datasets used by the RAG pipeline.
|-- src       #Main source code containing all modular components of the ingestion→retrieval→generation pipeline.
    |-- ingestion  #Loads and preprocesses documents (PDFs, videos, GitHub files) into a unified format.
    |-- splitting  #Splits documents into chunks optimized for retrieval and LLM context usage.
    |-- embedding  #Generates vector embeddings for chunks using selected embedding models and store in Chroma.
    |-- retrieval  #Implements BM25 and vector search to retrieve the top-k relevant chunks,fusion and rerank.
    |-- generation #Builds prompts and sends them to an LLM to generate answers using retrieved context.
    |-- evaluation #Contains evaluation scripts to benchmark retrieval/generation quality.
|-- backend_api
|   `-- main.py    #FastAPI backend exposing /retrieve and /generate endpoints for the frontend.
|-- frontend
|   `-- app.py     #Streamlit UI for querying the system, showing retrieved documents, and generating answers.
|-- notebooks      #Jupyter notebooks for experimentation, debugging, and evaluation.
|-- poetry.lock    #Locked dependency versions ensuring reproducible environments.
|-- pyproject.toml #Project configuration file defining dependencies, scripts, and metadata for Poetry.
|-- .gitignore     #Specifies which files and folders Git should ignore (e.g., venvs, cache, data dumps).



