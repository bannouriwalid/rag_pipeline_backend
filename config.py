import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    KB = os.getenv("KB")
    EVAL = os.getenv("EVAL")
    # RAG pipeline settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE"))

    # Milvus db
    MILVUS_CLUSTER_ENDPOINT = (os.getenv("MILVUS_CLUSTER_ENDPOINT"))
    MILVUS_API_KEY = (os.getenv("MILVUS_API_KEY"))
    GRAPHS_COLLECTION_NAME = os.getenv("GRAPHS_COLLECTION_NAME")
    MILVUS_VECTOR_DIM = int(os.getenv("MILVUS_VECTOR_DIM"))
    MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE")

    # NVIDIA
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    TEXTUALIZATION_MODEL = os.getenv("TEXTUALIZATION_MODEL")
    GENERATION_MODEL = os.getenv("GENERATION_MODEL")
    EVALUATION_MODEL = os.getenv("EVALUATION_MODEL")
