from flask_sqlalchemy import SQLAlchemy
from config import Config
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

db = SQLAlchemy()


def milvus_connect(cluster_endpoint, api_key):
    milvus_client = MilvusClient(uri=cluster_endpoint, token=api_key)
    return milvus_client


def milvus_create_collection(client, collection_name):
    # Drop the collection if it already exists (for a clean slate)
    if collection_name in client.list_collections():
        client.drop_collection(collection_name)

    # Define the schema
    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="graph_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=Config.MILVUS_VECTOR_DIM
            ),
            FieldSchema(  # optional metadata field
                name="graph_idx",
                dtype=DataType.INT64
            )
        ],
        description="Mean-pooled SBERT embeddings of knowledge graphs"
    )

    index_params = client.prepare_index_params()
    index_params.add_index("embedding", index_type="IVF_FLAT", metric_type="COSINE", index_params={"nlist": 64})
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        consistency_level="Strong",
        index_params=index_params
    )


def milvus_search(client, collection_name, k, q_emb):
    search_result = client.search(
        collection_name=collection_name,
        data=[q_emb.tolist()],
        limit=k,
        search_params={"metric_type": Config.MILVUS_METRIC_TYPE, "params": {}},
        output_fields=["graph_idx"]
    )
    return search_result


def milvus_insert_collection(client, collection_name, data):
    client.insert(
        collection_name=collection_name,
        data=data,
        auto_id=False
    )
    client.flush(collection_name)

