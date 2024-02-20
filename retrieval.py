from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore

from llama_index.core.vector_stores import VectorStoreQuery

from llama_index.core.schema import NodeWithScore
from typing import Optional

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine

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

# get vector store
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,  # openai embedding dimension
)

# 1. Generate a query embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
query_str = "Can you tell me about the key concepts for safety finetuning"
query_embedding = embed_model.get_query_embedding(query_str)

# 2. Query the vector datbase
# construct vector store query

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

# returns a VectorStoreQueryResult
query_result = vector_store.query(vector_store_query)
print(query_result.nodes[0].get_content())

# 3. Parse result into a set of nodes
nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))

# 4. Put into a retriever
class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

# 5. Plug this into a RetrieverQueryEngine to synthesize a response

# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
query_str = "How does Llama 2 perform compared to other open-source models?"
response = query_engine.query(query_str)

print(str(response))

print(response.source_nodes[0].get_content())