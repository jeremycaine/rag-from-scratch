from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.schema import TextNode

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore


# 1. load data
loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")
print(len(documents))

# 2. split documents
text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# 3. manually construct nodes
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

# 4. generate embeddings for each node
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

# 5. load nodes into vector store
db_name = "vector_db"
host = "localhost"
password = "password"
port = "5432"
user = "acme"

conn = psycopg2.connect(
    dbname=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,  # openai embedding dimension
)

# embeddings are stored in Pg vector store
vector_store.add(nodes)
print(vector_store)

