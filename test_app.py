import pytest
import pinecone
import os
import importlib
from dotenv import load_dotenv
load_dotenv()

if os.getenv("MODEL_NAME") == "huggingface":
    dimensions = 384
else:
    dimensions = 1536


def reset_pinecone() -> None:
    assert os.getenv("PINECONE_API_KEY") is not None
    assert os.getenv("PINECONE_ENV") is not None

    import pinecone

    importlib.reload(pinecone)

    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV"),
    )


def test_db():
    # initialize pinecone
    reset_pinecone()
    index = pinecone.Index(os.getenv("PINECONE_INDEX"))
    index_stats = index.describe_index_stats()
    assert index_stats["total_vector_count"] != 0
    assert index_stats["dimension"] == dimensions
    if index_stats["namespaces"].get(os.getenv("NAME_SPACE")) is not None:
        assert index_stats["namespaces"].get(os.getenv("NAME_SPACE"))["vector_count"] != 0
