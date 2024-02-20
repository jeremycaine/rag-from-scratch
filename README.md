# rag-from-scratch
Building RAG from scratch with open-source only with Llamaindex outlined [here](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html)

## 0. Environment Setup
In a Python 3.11 environment
```
conda create -n rag-from-scratch python=3.11
conda activate rag-from-scratch
pip install -r requirements.txt
```

### Sentence Transformers
```
# sentence transformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
```
Run `python setup-sentence-transformers.py` to check embedding model can be accessed.

### Postgres
Install Pg so you can run it locally, example [here](https://github.com/jeremycaine/setup-notes/tree/main/postgres)

Install [`pgvector`](https://github.com/pgvector/pgvector)
```
brew install pgvector
```

```
psgl postgres
CREATE ROLE acme WITH LOGIN PASSWORD 'password';
ALTER ROLE acme SUPERUSER;
CREATE EXTENSION vector;
DROP DATABASE vector_db;
CREATE DATABASE vector_db;
```
Run `python setup-connect-vector-store.py` to check database connection and vector store can be created.

### Llama2

## 1. Ingestion
Run `python ingestion.py`

Check vectors have been put in table
```
psql vector_db
> select * from data_llama2_paper;
```







