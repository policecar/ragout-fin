# Ragout Fin

A simple retrieval-augmented generation system that lets you download html files, chunk and embed them, then query against your vector index.

## Setup

```shell
uv venv
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt
```

Copy `example.env` to `.env` and set one of the API KEYs.  
Adapt `embed_model` and `llm_model` in `config.yml` accordingly.

Download the html files you want to embed and place them in `data/rag/`.


## Usage

```shell
# Start Neo4j
docker compose up -d

# Index documents
python rag.py --config config.yml --mode ingest --html_dir data/rag/

# Query our RAG
python rag.py --config config.yml --mode query --query "How do I deploy a Node.js application in a development environment?"
```
