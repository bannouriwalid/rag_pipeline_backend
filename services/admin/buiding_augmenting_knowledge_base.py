import json
import shutil
import time
from collections import defaultdict
import os
import re
import networkx as nx
from openai import OpenAI
import csv
from config import Config
from services.admin.scrapping import initial_scrapping
from services.embedding_models import load_model, load_text2embedding
import pandas as pd
import torch
from torch_geometric.data import Data
from db import milvus_connect, milvus_insert_collection, milvus_create_collection
from tqdm import tqdm
raw_files_folder = f'{Config.KB}/raw_files'
spo_folder = f'{Config.KB}/spo'
raw_spo_folder = spo_folder+'/raw'
normalized_folder = spo_folder+'/normalized&deduplicated'
refined_folder = spo_folder+'/refined'
path_nodes = f'{Config.KB}/nodes'
path_edges = f'{Config.KB}/edges'
path_graphs = f'{Config.KB}/graphs'
path_graphs_json = f'{Config.KB}/graphs_json'
added_files_folder = f'{Config.KB}/added_textfiles'


def textualize_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def format_dict(d):
        lines = []
        for key, value in d.items():
            if key == 'url':
                continue
            if isinstance(value, dict):
                lines.append(f"{key}:\n{format_dict(value)}")
            else:
                lines.append(f"{key}:\n{value}")
        return '\n'.join(lines)

    if isinstance(data, dict):
        text = format_dict(data)
    elif isinstance(data, list):
        text = '\n\n'.join(format_dict(item) for item in data)
    else:
        text = str(data)

    return text


def format_llm_output(output_text, save_path):
    lines = output_text.strip().splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("subject") or line.count(',') != 2:
            continue
        parts = [part.strip() for part in line.split(',')]
        if all(parts):
            cleaned_lines.append(parts)

    if not cleaned_lines:
        print(f"Warning: No valid SPO triples found in output: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object"])
        writer.writerows(cleaned_lines)


# Function to save LLM output text into cleaned CSV
def save_llm_output(file_name, output_text, output_path):
    file_path = os.path.join(output_path, f"{file_name}.csv")
    format_llm_output(output_text, file_path)


def count_spos_per_disease(spo_folder_path):
    spo_dict = defaultdict(list)

    for file_name in os.listdir(spo_folder_path):
        if file_name.endswith('.csv'):
            disease = file_name.split('_')[0]
            file_path = os.path.join(spo_folder_path, file_name)

            try:
                df = pd.read_csv(file_path)
                if {'subject', 'predicate', 'object'}.issubset(df.columns):
                    for _, row in df.iterrows():
                        spo_dict[disease].append((row["subject"], row["predicate"], row["object"]))
                else:
                    print(f"Skipped file (missing columns): {file_name}")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

    diseases = []
    counts = []
    # Optional: Print summary
    for disease, spo_list in sorted(spo_dict.items()):
        diseases.append(disease)
        counts.append(len(spo_list))
        print(f"{disease}: {len(spo_list)} SPO triples")

    total_spo = sum(len(v) for v in spo_dict.values())
    print(f"\nTotal SPO triples: {total_spo}")
    print(f"Total diseases: {len(spo_dict)}")

    return dict(spo_dict), diseases, counts


def spo_extraction(raw_data_path, output_path, single_file=False):
    # System prompt
    system_prompt = """
    You are a medical information-extraction assistant focused on respiratory diseases.
    Given a passage of text about {disease}, extract Subject-Predicate-Object (SPO) triples.

    Extraction rules:
    ────────────────
    • Extract any medically meaningful relation related to {disease} and its context.
    • Subjects and objects can be: symptoms, causes, treatments, diagnoses, risk factors, complications, or related medical concepts.
    • Use only these predicates (verbatim, lowercase):
      has_symptom, is_diagnosed_by, is_treated_by, is_caused_by, 
      is_prevented_by, has_risk_factor, leads_to_complication, 
      treats, causes, occurs_with, associated_with
    • Subjects/objects should be full terms — no pronouns or abbreviations.

    Output format:
    ─────────────
    • CSV only. First line must be:
      subject,predicate,object
    • No explanation, no blank lines, no quotation marks, no punctuation at end of terms.
    """

    # User prompt template (for formatting with each chunk)
    user_prompt_template = """
    EXTRACT SPO TRIPLES

    Disease context: {disease}

    Text:
    \"\"\"{text_chunk}\"\"\"

    Your response must be a CSV table starting with the header:
    subject,predicate,object
    """

    client = OpenAI(
        api_key=Config.NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    if not single_file:
        failed_files = 0
        folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
        os.makedirs(output_path, exist_ok=True)

        for folder_name in tqdm(folders, desc="Processing folders"):
            folder_path = os.path.join(raw_data_path, folder_name)
            files = os.listdir(folder_path)

            for file_name in tqdm(files, desc=f"→ {folder_name}", leave=False):
                file_path = os.path.join(folder_path, file_name)
                try:
                    raw_text = textualize_json(file_path)
                    completion = client.chat.completions.create(
                        model=Config.GENERATION_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt.format(disease=folder_name)},
                            {"role": "user", "content": user_prompt_template.format(text_chunk=raw_text, disease=folder_name)}
                        ],
                        temperature=0.2,
                        top_p=1.0,
                        max_tokens=2048,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stream=False
                    )

                    output_text = completion.choices[0].message.content
                    new_file_name = f"{folder_name}_{os.path.splitext(file_name)[0]}"
                    save_llm_output(new_file_name, output_text, output_path)

                except Exception as e:
                    print(f"Error processing file {file_name} in folder {folder_name}: {e}")
                    failed_files += 1

        all_extracted_triples, diseases, original_counts = count_spos_per_disease(output_path)
        return all_extracted_triples, diseases, original_counts
    else:
        file_name = os.path.basename(raw_data_path).split(".")[0]
        try:
            with open(raw_data_path, 'r', encoding='utf-8') as file:
                text = file.read()
            completion = client.chat.completions.create(
                model=Config.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt.format(disease=file_name)},
                    {"role": "user",
                     "content": user_prompt_template.format(text_chunk=text, disease=file_name)}
                ],
                temperature=0.2,
                top_p=1.0,
                max_tokens=2048,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )

            output_text = completion.choices[0].message.content
            save_llm_output(file_name, output_text, output_path)
            df = pd.read_csv(os.path.join(output_path, f"{file_name}.csv"))
            if {'subject', 'predicate', 'object'}.issubset(df.columns):
                triples = [(row['subject'], row['predicate'], row['object']) for _, row in df.iterrows()]
                return triples
            else:
                raise ValueError("CSV file must contain 'subject', 'predicate', and 'object' columns.")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")


def normalize_and_deduplicate(all_extracted_triples, diseases):
    normalized_spo_by_disease = {}
    dedup_stats = {}

    for disease in diseases:
        triples = all_extracted_triples[disease]

        normalized_triples = []
        seen_triples = set()
        empty_removed_count = 0
        duplicates_removed_count = 0

        for i, (s, p, o) in enumerate(triples):
            s = s.strip().lower() if isinstance(s, str) else ''
            p = re.sub(r'\s+', ' ', p.strip().lower()) if isinstance(p, str) else ''
            o = o.strip().lower() if isinstance(o, str) else ''

            if all([s, p, o]):
                key = (s, p, o)
                if key not in seen_triples:
                    normalized_triples.append({'subject': s, 'predicate': p, 'object': o})
                    seen_triples.add(key)
                else:
                    duplicates_removed_count += 1
            else:
                empty_removed_count += 1

        normalized_spo_by_disease[disease] = normalized_triples
        dedup_stats[disease] = {
            "original": len(triples),
            "kept": len(normalized_triples),
            "duplicates_removed": duplicates_removed_count,
            "empty_removed": empty_removed_count
        }
    # Summary
    print("Deduplication Summary Per Disease:")
    for disease in sorted(dedup_stats):
        stats = dedup_stats[disease]
        print(
            f"{disease}: Kept {stats['kept']} / {stats['original']} | Duplicates: {stats['duplicates_removed']}, Empty: {stats['empty_removed']}")
    return normalized_spo_by_disease


def export_spo_per_disease(normalized_spo_by_disease, path):
    os.makedirs(path, exist_ok=True)
    for disease, triples in normalized_spo_by_disease.items():
        filename = f"{disease.lower().replace(' ', '_')}.csv"
        file_path = os.path.join(path, filename)

        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["subject", "predicate", "object"])
            writer.writeheader()
            writer.writerows(triples)
    print(f"Exported SPO CSV files to: {path}")


def read_csv_rows(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def rows_to_csv_string(rows):
    output = ["subject,predicate,object"]
    for row in rows:
        output.append(f"{row['subject']},{row['predicate']},{row['object']}")
    return "\n".join(output)


def adaptive_chunks(rows, max_chars):
    """Yield batches of rows such that the total character length of the CSV string stays below max_chars."""
    batch = []
    current_length = len("subject,predicate,object\n")  # header

    for row in rows:
        row_str = f"{row['subject']},{row['predicate']},{row['object']}\n"
        if current_length + len(row_str) > max_chars:
            if batch:
                yield batch
                batch = []
                current_length = len("subject,predicate,object\n")
        batch.append(row)
        current_length += len(row_str)

    if batch:
        yield batch


def refine_spo_from_csv(input_path, output_path, max_chars_per_batch=5000):
    refining_system_prompt = """
    You are a medical SPO-triple refinement assistant for respiratory-disease knowledge graphs.

    Task
    ────
    Given SPO triples about the disease {disease}, return a single, clean, fully-connected graph:

    • Eliminate semantic duplicates.  
    • Clarify vague terms (use formal medical wording, no abbreviations or pronouns).  
    • Ensure every node belongs to ONE connected component anchored on {disease}.  
      – If a node is isolated but clearly relates to {disease}, ADD one factual triple to link it.  
      – If connection is uncertain, leave the triple unchanged (do NOT fabricate facts).  
    • All text must be lowercase.

    Output format
    ─────────────
    csv only — first line literally:
    subject,predicate,object
    No explanations, blank lines, quotes, or trailing punctuation.
    """

    refining_user_prompt_template = """
    REFINE SPO TRIPLES  –  DISEASE: {disease}

    Below is a csv table of extracted triples.

    Your tasks
    ──────────
    1. Deduplicate rows that convey the same fact.  
    2. Rephrase unclear wording for precision.  
    3. Ensure every node appears in at least one edge connected (directly or indirectly) to "{disease}".  
    4. If a node is isolated yet obviously related, add ONE triple using an allowed predicate to connect it.  
    5. Do NOT invent new medical facts.  
    6. Return lowercase csv, starting with the header.

    Input triples:
    {csv_triples}

    Your output must start with:
    subject,predicate,object
    """

    client = OpenAI(
        api_key=Config.NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    os.makedirs(output_path, exist_ok=True)
    failed_files = 0

    for file_name in tqdm(os.listdir(input_path), desc="Refining SPO per disease"):
        if not file_name.endswith('.csv'):
            continue

        disease = os.path.splitext(file_name)[0].replace('_', ' ')
        file_path = os.path.join(input_path, file_name)

        try:
            rows = read_csv_rows(file_path)
            all_refined_rows = []

            for i, batch in enumerate(adaptive_chunks(rows, max_chars=max_chars_per_batch)):
                csv_chunk = rows_to_csv_string(batch)

                completion = client.chat.completions.create(
                    model=Config.GENERATION_MODEL,
                    messages=[
                        {"role": "system", "content": refining_system_prompt.format(disease=disease)},
                        {"role": "user", "content": refining_user_prompt_template.format(disease=disease, csv_triples=csv_chunk)}
                    ],
                    temperature=0.2,
                    top_p=1.0,
                    max_tokens=2048,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=False
                )

                output_text = completion.choices[0].message.content.strip()

                # Parse the refined triples
                lines = output_text.splitlines()
                for line in lines[1:]:  # Skip header
                    parts = line.split(",")
                    if len(parts) == 3:
                        subject, predicate, object_ = [p.strip() for p in parts]
                        all_refined_rows.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": object_
                        })

            # Deduplicate after merging batches
            unique_rows = [dict(t) for t in {tuple(d.items()) for d in all_refined_rows}]

            # Save final merged and deduplicated file
            out_filename = f"{file_name.replace('.csv', '')}_refined.csv"
            out_path = os.path.join(output_path, out_filename)

            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object"])
                writer.writeheader()
                writer.writerows(unique_rows)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            failed_files += 1

    print(f"\nCompleted refinement with {failed_files} failure(s).")


def summarize_refinement_process(raw_path, refined_path):
    summary = []

    # Helper function to count SPO triples from a folder
    def load_spo_counts(folder_path):
        spo_counts = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                disease_name = file_name.replace('_refined', '').replace('.csv', '').replace('_', ' ')
                file_path = os.path.join(folder_path, file_name)

                try:
                    df = pd.read_csv(file_path)
                    if {'subject', 'predicate', 'object'}.issubset(df.columns):
                        spo_counts[disease_name] = len(df)
                    else:
                        print(f"Skipped file (missing columns): {file_name}")
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")
        return spo_counts

    raw_counts = load_spo_counts(raw_path)
    refined_counts = load_spo_counts(refined_path)

    diseases = sorted(set(raw_counts.keys()).union(refined_counts.keys()))

    print(f"{'Disease':<50} {'Raw':>6} {'Refined':>8} {'Δ':>6}")
    print("-" * 52)

    total_raw = total_refined = 0

    for disease in diseases:
        raw = raw_counts.get(disease, 0)
        refined = refined_counts.get(disease, 0)
        delta = refined - raw
        total_raw += raw
        total_refined += refined

        print(f"{disease:<50} {raw:>6} {refined:>8} {delta:>6}")
        summary.append({
            "disease": disease,
            "raw": raw,
            "refined": refined,
            "delta": delta
        })

    print("-" * 52)
    print(f"{'Total':<50} {total_raw:>6} {total_refined:>8} {total_refined - total_raw:>6}")

    return summary


def csv_file_to_disease(file_name: str) -> str:
    return file_name.replace("_refined.csv", "").replace("_", " ")


def extract_graphs_from_directory(
    input_folder: str,
    output_json_path: str,
    verbose: bool = False,
):
    all_graphs = []  # list to store only the largest subgraph per file
    disease_stats = defaultdict(lambda: {"graphs": 0, "nodes": 0, "edges": 0})

    for file_name in tqdm(os.listdir(input_folder), desc="Building graphs"):
        if not file_name.endswith("_refined.csv"):
            continue

        disease = csv_file_to_disease(file_name)
        file_path = os.path.join(input_folder, file_name)

        rows = read_csv_rows(file_path)

        # Build the MultiDiGraph
        G = nx.MultiDiGraph()
        for row in rows:
            G.add_edge(row["subject"], row["object"], predicate=row["predicate"])

        # Find the largest weakly connected component
        largest_subgraph = None
        max_nodes = max_edges = -1

        for comp_nodes in nx.weakly_connected_components(G):
            sub = G.subgraph(comp_nodes).copy()  # make it a full graph copy
            n_nodes = sub.number_of_nodes()
            n_edges = sub.number_of_edges()

            if (n_nodes > max_nodes) or (n_nodes == max_nodes and n_edges > max_edges):
                largest_subgraph = sub
                max_nodes = n_nodes
                max_edges = n_edges

        # Save the largest subgraph (if any)
        if largest_subgraph is not None:
            triples = [
                [u, d["predicate"], v] for u, v, d in largest_subgraph.edges(data=True)
            ]

            all_graphs.append(triples)

            disease_stats[disease]["graphs"] = 1
            disease_stats[disease]["nodes"] = max_nodes
            disease_stats[disease]["edges"] = max_edges

            if verbose:
                print(
                    f"{disease:<45} | largest graph | "
                    f"{max_nodes:>3} nodes | {max_edges:>3} edges"
                )

    # Write JSON
    with open(output_json_path, "w", encoding="utf-8") as fp:
        json.dump(all_graphs, fp, indent=2, ensure_ascii=False)

    # Print summary
    print("\nSummary by disease")
    print(f"{'disease':<45} {'#graphs':>8} {'nodes':>10} {'edges':>10}")
    print("-" * 55)
    total_g = total_n = total_e = 0
    for dis, s in sorted(disease_stats.items()):
        print(
            f"{dis:<45} {s['graphs']:>8} {s['nodes']:>10} {s['edges']:>10}"
        )
        total_g += s["graphs"]
        total_n += s["nodes"]
        total_e += s["edges"]
    print("-" * 55)
    print(f"{'TOTAL':<45} {total_g:>8} {total_n:>10} {total_e:>10}")

    return all_graphs, disease_stats


def create_nodes_edges_folders():
    # Load the graphs from the JSON file
    with open(path_graphs_json + '/graphs.json', 'r') as f:
        graphs = json.load(f)

    # Create directories if they don't exist
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    # Process each graph
    for i, triples in enumerate(tqdm(graphs)):
        node_map = {}  # Maps node label → node ID
        edges = []

        for h, r, t in triples:
            h = h.lower()
            t = t.lower()
            if h not in node_map:
                node_map[h] = len(node_map)
            if t not in node_map:
                node_map[t] = len(node_map)
            edges.append({'src': node_map[h], 'edge_attr': r, 'dst': node_map[t]})

        # Convert node map to DataFrame
        nodes_df = pd.DataFrame(
            [{'node_id': v, 'node_attr': k} for k, v in node_map.items()],
            columns=['node_id', 'node_attr']
        )

        # Convert edge list to DataFrame
        edges_df = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        # Save to CSV
        nodes_df.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges_df.to_csv(f'{path_edges}/{i}.csv', index=False)


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


def build_initial_knowledge_base():
    # creates /raw_files with a folder per disease and many json files per disease
    initial_scrapping()
    time.sleep(5)
    # creates /spo
    os.makedirs(spo_folder, exist_ok=True)
    # creates /spo/raw csv for each file all diseases together
    all_extracted_triples, diseases, original_counts = spo_extraction(raw_files_folder, raw_spo_folder, False)
    time.sleep(5)
    # just normalize and deduplicate
    normalized_spo_by_disease = normalize_and_deduplicate(all_extracted_triples, diseases)
    # creates /spo/normalized&deduplicated one csv file per disease
    export_spo_per_disease(normalized_spo_by_disease, normalized_folder)
    time.sleep(5)
    # creates /spo/refined a refined csv file per disease
    refine_spo_from_csv(normalized_folder, refined_folder)
    time.sleep(5)
    # display a summary
    summarize_refinement_process(normalized_folder, refined_folder)
    # creates /spo/graphs.json containing list of graphs one graph per disease
    extract_graphs_from_directory(refined_folder, spo_folder+'/graphs.json', verbose=True)
    time.sleep(5)
    # creates /nodes and /edges one csv file per graph
    create_nodes_edges_folders()
    time.sleep(5)
    # creates /graphs one .pt file per graph embedd and store in milvus
    graph_embedding_store()


def add_new_disease(complete_file_path):
    os.makedirs(added_files_folder, exist_ok=True)
    file_name = os.path.basename(complete_file_path)
    # Define destination path
    destination = os.path.join(added_files_folder, file_name)
    # Copy the file to the destination folder
    shutil.copy(complete_file_path, destination)
    all_extracted_triples = spo_extraction(complete_file_path, raw_spo_folder, True)
    normalized_spo_by_disease = normalize_and_deduplicate({file_name.split(".")[0]: all_extracted_triples}, [file_name.split(".")[0]])
    export_spo_per_disease(normalized_spo_by_disease, normalized_folder)
    refine_spo_from_csv(normalized_folder, refined_folder)
    summarize_refinement_process(normalized_folder, refined_folder)
    extract_graphs_from_directory(refined_folder, spo_folder + '/graphs.json', verbose=True)
    create_nodes_edges_folders()
    graph_embedding_store()

"""
# Detailed Code Report: Building and Augmenting Knowledge Base

## Overview
This module implements a comprehensive pipeline for building and maintaining a medical knowledge base focused on respiratory diseases. The system processes raw medical data, extracts structured information, and creates a graph-based knowledge representation that can be queried and updated.

## Key Components

### 1. Data Processing Pipeline
- Raw data collection and organization
- SPO (Subject-Predicate-Object) triple extraction
- Normalization and deduplication
- Refinement and validation
- Graph construction and embedding

### 2. Main Functions

#### Data Collection and Initial Processing
- `initial_scrapping()`: Initiates data collection
- `textualize_json()`: Converts JSON data to text format
- `spo_extraction()`: Extracts SPO triples from text using LLM

#### Data Cleaning and Normalization
- `normalize_and_deduplicate()`: Removes duplicates and normalizes terms
- `refine_spo_from_csv()`: Further refines and validates SPO triples
- `summarize_refinement_process()`: Provides statistics on data refinement

#### Graph Construction and Storage
- `extract_graphs_from_directory()`: Creates graph representations
- `create_nodes_edges_folders()`: Organizes graph components
- `graph_embedding_store()`: Generates and stores graph embeddings

### 3. Data Flow
1. Raw Data → Text Conversion
2. Text → SPO Triples
3. SPO Triples → Normalized Triples
4. Normalized Triples → Refined Triples
5. Refined Triples → Graph Structure
6. Graph Structure → Vector Embeddings

### 4. Storage Structure
- `/raw_files`: Original data
- `/spo`: Subject-Predicate-Object triples
  - `/raw`: Initial extractions
  - `/normalized&deduplicated`: Cleaned data
  - `/refined`: Final validated triples
- `/nodes`: Graph node definitions
- `/edges`: Graph edge definitions
- `/graphs`: Graph embeddings
- `/graphs_json`: Graph representations

### 5. Key Features
- Automated data processing pipeline
- LLM-based information extraction
- Graph-based knowledge representation
- Vector embeddings for similarity search
- Milvus integration for vector storage
- Support for incremental updates

### 6. Dependencies
- OpenAI API for LLM processing
- NetworkX for graph operations
- PyTorch for embeddings
- Milvus for vector storage
- Pandas for data manipulation

### 7. Error Handling
- Comprehensive error checking in data processing
- Validation at each pipeline stage
- Graceful handling of missing or malformed data

### 8. Performance Considerations
- Batch processing for large datasets
- Efficient graph construction
- Optimized embedding generation
- Vectorized operations where possible

### 9. Maintenance and Updates
- Support for adding new diseases
- Incremental knowledge base updates
- Data validation and quality checks

### 10. Limitations and Future Improvements
- Dependency on external LLM API
- Potential for information loss in normalization
- Limited to respiratory diseases
- Could benefit from parallel processing
- Potential for enhanced validation rules

## Usage Examples
1. Building initial knowledge base:
   ```python
   build_initial_knowledge_base()
   ```

2. Adding new disease data:
   ```python
   add_new_disease("path/to/disease_data.json")
   ```

## Security Considerations
- API key management
- Data privacy in medical information
- Secure storage of embeddings

## Best Practices
- Regular data validation
- Backup of processed data
- Monitoring of API usage
- Documentation of changes
"""
