from config import Config
from services.pcst import retrieval_via_pcst
from services.embedding_models import load_model, load_text2embedding
from db import milvus_connect, milvus_search
import torch
import os
import pandas as pd

path_nodes = f'{Config.KB}/nodes'
path_edges = f'{Config.KB}/edges'
path_graphs = f'{Config.KB}/graphs'


def retrieval(question, k=1):
    model, tokenizer, device = load_model()
    text2embedding = load_text2embedding
    # Encode question
    q_emb = text2embedding(model, tokenizer, device, [question])[0]

    # Ensure collection is loaded before search
    milvus_client = milvus_connect(Config.MILVUS_CLUSTER_ENDPOINT, Config.MILVUS_API_KEY)
    try:
        milvus_client.load_collection(Config.GRAPHS_COLLECTION_NAME)
    except Exception as e:
        print(f"Error loading collection: {e}")
        return [], []

    # Perform similarity search in Milvus
    search_results = milvus_search(milvus_client, Config.GRAPHS_COLLECTION_NAME, k, q_emb)

    # Extract graph indices from results
    hits = search_results[0]
    graph_indices = [hit["entity"]["graph_idx"] for hit in hits]

    # Collect sub_graphs and descriptions
    sub_graphs = []
    descriptions = []

    for index in graph_indices:
        nodes_path = f'{path_nodes}/{index}.csv'
        edges_path = f'{path_edges}/{index}.csv'
        graph_path = f'{path_graphs}/{index}.pt'

        if not (os.path.exists(nodes_path) and os.path.exists(edges_path) and os.path.exists(graph_path)):
            print(f"Missing data for graph {index}")
            continue

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        if len(nodes) == 0:
            print(f"Empty graph at index {index}")
            continue

        graph = torch.load(graph_path)

        # Apply retrieval logic
        sub_g, desc = retrieval_via_pcst(
            graph=graph,
            q_emb=q_emb,
            textual_nodes=nodes,
            textual_edges=edges,
            topk=3,
            topk_e=5,
            cost_e=0.5
        )

        sub_graphs.append(sub_g)
        descriptions.append(desc)

    return sub_graphs, descriptions
