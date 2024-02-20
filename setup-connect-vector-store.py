import psycopg2

from llama_index.vector_stores.postgres import PGVectorStore

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
print(vector_store)