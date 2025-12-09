# JASP RAG Protocol
A multimodal Retrieval-Augmented Generation (RAG) system designed for the JASP statistical software ecosystem.

It supports PDF manuals, GitHub documentation, YouTube tutorials, and stores all content in ChromaDB for efficient keyword + semantic retrieval.

Users can query the system through a FastAPI backend and an intuitive Streamlit frontend.

# Features
- Multimodal ingestion (PDF → Markdown → JSON, GitHub markdown, YouTube transcripts)
- utomated image extraction + description from PDF pages
- Token-based splitting for consistent chunk sizes
- BM25, vector, hybrid fusion, and reranked retrieval modes
- FastAPI backend for retrieval & generation
- Streamlit UI for interactive exploration
- Fully reproducible setup via Poetry

# Prerequisites
- Python 3.11 or 3.12 (required by pyproject.toml)
- Poetry 2.x for dependency management  
  Install: https://python-poetry.org/docs/#installation
- Ollama (Required)
       Used for all LLM and VLM inference (Mistral, LLaMA, LLaVA).
       Install: https://ollama.com  
       Start the service: 'ollama serve'
- FFmpeg (Required for Whisper + video pipelines)
       macOS:   brew install ffmpeg  
       Linux:   sudo apt-get install ffmpeg  
       Windows: https://www.gyan.dev/ffmpeg/builds/
- Poppler (Required for pdf2image on some systems).
       Only needed if your machine fails on PDF → image conversion.
       macOS: brew install poppler  
       Linux: sudo apt-get install poppler-utils  
       Windows binaries available online


# Quick Start
0. Install dependencies:
       ' poetry install'

        Poetry will automatically installs:
              FastAPI
              Streamlit
              LlamaIndex ecosystem
              BM25 retriever
              FlagEmbedding reranker
              ChromaDB
              Whisper
              YouTube processing libraries
              PDF processing tools
              All custom JASP RAG dependencies

1. Start the FastAPI backend in a terminal:
       poetry run uvicorn backend.main:app --reload --port 8000

2. Start this Streamlit frontend in a seperate terminal to open the app in your brower:
       poetry run streamlit run frontend/app.py

3. Stop the app:
       Press CTRL+C in the terminals.

Example:
- Type your question in the text area (e.g., "How do I run ANOVA in JASP?")
- Click "Find helpful Docs" → the system queries the backend and finds best match documents in the knowledge base.
- Choose model (default: mistral:7b).Click"Generate answer"
       → the system queries the backend and answer questions with reference.

# Example Workflow
1. Type a question such as:
"How do I run ANOVA in JASP?"

2. Click "Find helpful docs"
→ The system retrieves relevant PDF sections, GitHub docs, and video transcripts

3. Optionally choose an LLM (default: mistral:7b)

4. Click "Generate Answer"
→ The backend uses retrieved context to generate a reference-supported answer

# Structure
|-- README.md #High-level overview of the project, setup instructions, and how everything works.

|-- data      #Stores raw, processed, and intermediate datasets used by the RAG pipeline.

|-- src       #Main source code containing all modular components of the ingestion→retrieval→generation pipeline.

    |-- ingestion  #step 1: The ingestion layer prepares all external JASP resources 
                            (PDF manuals, GitHub documentation, and YouTube tutorials) into a clean, structured, and RAG-ready format before splitting.Each ingestion script performs a specific transformation and saves outputs into the data/processed/ directory, where they can be splitted for embedding.
                            💙 pdf ingestion: The PDF ingestion for the JASP manuals consists of **two scripts**:
                                   1. `src/ingestion/pdf_text_loader.py`  
                                   - Downloads PDFs listed in `data/raw_pdf/pdf_list.json`  
                                   - Cleans page text, detects section titles, normalizes equations/tables  
                                   - Converts each PDF → cleaned Markdown → section-level JSON in  
                                   `data/processed/pdf/text/`
                                   2. 'src/ingestion/pdf_text_add_image_description.py'
                                   - Extracts images from each PDF page
                                   - Summarizes them with an Ollama/LLaVA model
                                   - Injects image paths + short descriptions into the section JSON
                                   - Recomputes token_length and writes final enriched JSON to 
                                   'data/processed/pdf/final_enriched/'

                            To add more PDFs to the pdf ingestion pipeline:
                                   1. **Add the PDF entry to `data/raw_pdf/pdf_list.json`**
                                   2. Run the text ingestion pipeline to download the new PDF and converts it to Markdown + section JSON (details need to revise to match your specific pdf case)
                                   3. Run the image-description enrichment pipeline to extracts images, summarizes them, and inserts them into section JSON
                                   4.split as usual :The new enriched JSON will automatically be included when you run spilitting script for the RAG system.


                            💙 Github mardown ingestion: The ingestion is built inside 
                                   `src/ingestion/github_md_loader.py`:
                                   - Downloads all `.md` files inside a specific folder of a GitHub repo  
                                   - Stores them in `data/raw_github/` 

                            To add additional GitHub repos or folders:  
                                   1. Edit `data/raw_github/github_list.json`. Add a new entry under the `"github"` list.  
                                   2. Run the GitHub loader `src/ingestion/github_md_loader.py`
                                   3. Proceed with spilitting as usual

                            💙 YouTube video ingestion: implemented in `src/ingestion/video_transcript_loader.py`:
                                   1. Read a list of videos from `data/raw_video/video.json`
                                   2. Save one JSON per video to:'data/processed/video/'

                            Adding more videos to the ingestion pipeline:
                                   1.define new Video sources in `data/raw_video/video.json`.
                                   2. run the video loader `src/ingestion/video_transcript_loader.py`
                                   3.3. Proceed with spilitting as usual

    |-- splitting  #step 2: The src/splitting folder contains the final preprocessing 
                            stage that  converts all enriched content—PDF manual sections, GitHub Markdown files, and YouTube video transcripts—into consistent, token-bounded chunks used for embedding and retrieval. 
                            
                            Each splitter (PDF, GitHub, and video) loads its respective processed input files, applies semantic token-based splitting to ensure chunks stay around 500 tokens, preserves important metadata such as page numbers, chapter titles, timestamps, and source URLs, and outputs a unified JSON structure (data/processed/chunks/*.json). 
                            
                            These chunked files form the complete, retrieval-ready document set for BM25 and for embedding.

    |-- embedding  #Step 3: After all data sources have been ingested 
                            and split into token-bounded chunks, the embedding pipeline converts each chunk into a vector representation and stores it in a persistent ChromaDB database.

    |-- retrieval  #step 4:The retrieval layer combines **BM25 keyword search**, 
                            **semantic vector search**,  and optional **cross-encoder reranking** to return the most relevant docs

    |-- generation #step 5: Once retrieval is configured and the vector store is built, 
                            the generation pipeline produces the final answer for the user.


|-- backend_api

       |-- main.py    #step 6: The backend lives in `backend_api/main.py` 
                            and exposes a small REST API for retrieval, RAG generation, and PDF previews.

|-- frontend

       |-- app.py     #step 7: The user interface lives in 'frontend/app.py',
                            It connects to the   FastAPI backend and provides an interactive way to: type a question and run retrieval (/retrieve),inspect the ranked sources (PDF, GitHub, video), preview original documents, optionally generate an LLM answer via Ollama

|-- poetry.lock    #Locked dependency versions ensuring reproducible environments.

|-- pyproject.toml #Project configuration file defining dependencies, scripts, and metadata for Poetry.

|-- .gitignore     #Specifies which files and folders Git should ignore (e.g., venvs, cache, data dumps).



# update: When you make changes in VS Code:
git add .
git commit -m "full pipeline version-7.06"
git push

