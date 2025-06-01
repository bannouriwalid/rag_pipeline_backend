import json
from services.embedding_models import load_model, load_text2embedding
import os
from config import Config
import pandas as pd
import torch
from torch_geometric.data import Data
from db import milvus_connect, milvus_insert_collection, milvus_create_collection
from tqdm import tqdm
path_nodes = f'{Config.KB}/nodes'
path_edges = f'{Config.KB}/edges'
path_graphs = f'{Config.KB}/graphs'
path_graphs_json = f'{Config.KB}/graphs_json'


def graph_embedding_store():
    print("Embedding and storing graphs in milvusDB...")
    with open(path_graphs_json + '/graphs.json', 'r') as f:
        graphs = json.load(f)

    model, tokenizer, device = load_model()
    text2embedding = load_text2embedding

    os.makedirs(path_graphs, exist_ok=True)

    milvus_vectors = []

    for index in tqdm(range(len(graphs))):
        # --- Load nodes & edges ---
        nodes_path = f'{path_nodes}/{index}.csv'
        edges_path = f'{path_edges}/{index}.csv'
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            print(f'Skipping graph {index} (missing files)')
            continue

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        nodes.fillna({"node_attr": ""}, inplace=True)

        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue

        # --- Embed node and edge attributes ---
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])
        # --- Save graph as torch_geometric.Data ---
        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')
        # --- Compute graph-level embedding (mean of node embeddings) ---
        graph_embedding = torch.mean(x, dim=0).cpu().tolist()

        # --- Store in Milvus format: [graph_id, embedding, index] ---
        milvus_vectors.append({"graph_id": index, "embedding": graph_embedding, "graph_idx": index})

    print("storing in milvus")
    # --- Final batch insert into Milvus ---
    if milvus_vectors:
        milvus_client = milvus_connect(Config.MILVUS_CLUSTER_ENDPOINT, Config.MILVUS_API_KEY)
        milvus_create_collection(milvus_client, Config.GRAPHS_COLLECTION_NAME)
        milvus_insert_collection(milvus_client, Config.GRAPHS_COLLECTION_NAME, milvus_vectors)
        print(f"Inserted {len(milvus_vectors)} graph embeddings into Milvus.")
    else:
        print("No graphs were inserted into Milvus.")
