import json
import re
import time
from pathlib import Path
from services.admin.bechmarking import generate_qa_pairs, load_qa, group_sources, load_files
from dotenv import dotenv_values, set_key, load_dotenv
from config import Config
import networkx as nx
import os
from services.admin.buiding_augmenting_knowledge_base import build_initial_knowledge_base, add_new_disease
from services.user.evaluation import evaluate_response
from services.user.response_generation import generate_response
from services.user.retrieval import retrieval
from datetime import datetime
from sqlalchemy import func
from models import User, Response, ValidatedResponse, Evaluation
from db import db
ENV_FILE = ".env"
qa_folder = f'{Config.EVAL}/generated_qa'
raw_json_files_folder = f'{Config.KB}/raw_files'
raw_text_files_folder = f'{Config.KB}/added_textfiles'
evaluation_folder = f'{Config.EVAL}'
spo_folder = f'{Config.KB}/spo'

def recreate_initial_state():
    build_initial_knowledge_base()


def add_new_text_file(path):
    add_new_disease(path)


def get_settings():
    config = dotenv_values(ENV_FILE)
    return config


def update_settings(new_settings: dict):
    if not new_settings:
        return {"error": "No data provided"}

    # Update keys in the .env file
    for key, value in new_settings.items():
        set_key(ENV_FILE, key, str(value))

    # Reload environment variables in the current process
    load_dotenv(override=True)
    return dotenv_values(ENV_FILE)


def trigger_evaluation(generate_qa):
    group_sources(raw_json_files_folder)
    if generate_qa:
        os.makedirs(qa_folder, exist_ok=True)
        # Load all JSON files
        files = load_files(raw_json_files_folder, raw_text_files_folder)
        # Process each file
        for file_data in files:
            file_path = file_data['file_path']
            content = file_data['content']
            file_type = file_data['type']
            # Generate Q&A pairs
            qa_pairs = generate_qa_pairs(content, file_type)

            if qa_pairs:
                # Create output filename
                relative_path = os.path.relpath(file_path, raw_json_files_folder)
                output_filename = os.path.join(qa_folder, f"{Path(relative_path).stem}_qa.json")

                # Save Q&A pairs
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, indent=2)
                print(f"Saved Q&A pairs to {output_filename}")

            # Add a small delay to avoid rate limiting
            time.sleep(10)

    all_questions, all_answers = load_qa(qa_folder)
    average_scores = {
        "clarity_score": 0,
        "exactitude_score": 0,
        "context_adherence_score": 0,
        "relevance_score": 0,
        "completeness_score": 0,
        "logical_flow_score": 0,
        "uncertainty_handling_score": 0,
        "overall_feedback": 0
    }
    current_date = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    file_name = evaluation_folder+f"/evaluation_{current_date}.txt"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("===== Evaluation Report =====\n")
        f.write(f"Generated on: {current_date}\n")
        f.write(f"Number of questions evaluated: {len(all_questions)}\n\n")
        f.write("This report presents the automatic evaluation of generated answers across multiple criteria:\n")
        f.write(
            "clarity, exactitude, adherence to context, relevance, completeness, logical flow, and handling of uncertainty.\n\n")
        f.write("\n===== Questions answers detail=====\n")

    for i, question in enumerate(all_questions[:2]):  # remove this later but it takes too much time
        sub_graphs, descriptions = retrieval(question, 1)
        answer = generate_response(descriptions, question, False)
        scores = evaluate_response(question, descriptions, answer, file_name, all_answers[i], True)
        for key in average_scores.keys():
            value = scores[key]
            if isinstance(value, list):
                value = value[0] if value else 0  # safe fallback
            average_scores[key] += float(value)
        time.sleep(5)

    for key in average_scores.keys():
        average_scores[key] /= len(all_questions)

    with open(file_name, "a", encoding="utf-8") as f:
        f.write("\n===== Summary of Average Scores =====\n")
        for key, value in average_scores.items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")

    return average_scores


def get_kb_overview():
    kb_file = spo_folder+"/graphs.json"
    with open(kb_file, "r", encoding="utf-8") as f:
        all_graphs = json.load(f)

    overview = {
        "total_graphs": len(all_graphs),
        "graphs": []
    }

    for i, triples in enumerate(all_graphs):
        G = nx.DiGraph()
        for h, r, t in triples:
            G.add_edge(h, t, relation=r)

        overview["graphs"].append({
            "graph_index": i,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges()
        })

    return overview


def get_overview():
    # --- User Roles Count ---
    total_patients = db.session.query(User).filter_by(role='patient').count()
    total_doctors = db.session.query(User).filter_by(role='doctor').count()

    # --- Answer Stats ---
    total_answers = db.session.query(Response).count()
    total_validated = db.session.query(Response).filter_by(status="validated").count()
    total_not_validated = db.session.query(Response).count() - total_validated

    validated_percent = round((total_validated / total_answers) * 100, 2) if total_answers else 0
    not_validated_percent = round((total_not_validated / total_answers) * 100, 2) if total_answers else 0

    # --- Confirmed & Corrected Responses ---
    nb_confirmed = db.session.query(ValidatedResponse).filter_by(confirmed=True).count()
    nb_corrected = db.session.query(ValidatedResponse).count() - nb_confirmed

    confirmed_percent = round((nb_confirmed / total_validated) * 100, 2) if total_answers else 0
    corrected_percent = round((nb_corrected / total_validated) * 100, 2) if total_answers else 0

    # --- Evaluation Stats ---
    total_evaluations = db.session.query(Evaluation).count()

    avg_scores = db.session.query(
        func.avg(Evaluation.clarity_score).label("clarity"),
        func.avg(Evaluation.exactitude_score).label("exactitude"),
        func.avg(Evaluation.context_adherence_score).label("context"),
        func.avg(Evaluation.relevance_score).label("relevance"),
        func.avg(Evaluation.completeness_score).label("completeness"),
        func.avg(Evaluation.logical_flow_score).label("flow"),
        func.avg(Evaluation.uncertainty_handling_score).label("uncertainty"),
        func.avg(Evaluation.overall_feedback).label("overall"),
    ).first()

    # --- Pipeline Evaluation Files ---
    eval_files = [f for f in os.listdir(evaluation_folder) if f.startswith("evaluation_") and f.endswith(".txt")]
    eval_files.sort(reverse=True)  # latest first

    nb_pipeline_evals = len(eval_files)
    last_eval_file = eval_files[0] if eval_files else None
    last_eval = extract_human_readable_date(last_eval_file)
    # --- Added Text Files (in another folder) ---
    nb_added_text_files = len([f for f in os.listdir(raw_text_files_folder) if f.endswith(".txt")])

    return {
        "total_patients": total_patients,
        "total_doctors": total_doctors,
        "total_answers": total_answers,
        "total_validated": total_validated,
        "total_not_validated": total_not_validated,
        "validated_percent": validated_percent,
        "not_validated_percent": not_validated_percent,
        "total_evaluations": total_evaluations,
        "average_scores": {
            "clarity": round(avg_scores.clarity or 0, 2),
            "exactitude": round(avg_scores.exactitude or 0, 2),
            "context": round(avg_scores.context or 0, 2),
            "relevance": round(avg_scores.relevance or 0, 2),
            "completeness": round(avg_scores.completeness or 0, 2),
            "flow": round(avg_scores.flow or 0, 2),
            "uncertainty": round(avg_scores.uncertainty or 0, 2),
            "overall": round(avg_scores.overall or 0, 2),
        },
        "nb_confirmed": nb_confirmed,
        "nb_corrected": nb_corrected,
        "confirmed_percent": confirmed_percent,
        "corrected_percent": corrected_percent,
        "nb_pipeline_evaluations": nb_pipeline_evals,
        "last_evaluation_date": last_eval,
        "nb_added_text_files": nb_added_text_files
    }


def extract_human_readable_date(filename):
    match = re.search(r"evaluation_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.txt", filename)
    if not match:
        return "Unknown date"

    date_part, hour, minute, second = match.groups()
    dt_string = f"{date_part} {hour}:{minute}:{second}"
    dt_obj = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")

    return dt_obj.strftime("%B %#d, %Y at %H:%M:%S")
