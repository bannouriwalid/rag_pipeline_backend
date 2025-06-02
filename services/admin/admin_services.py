import json
import time
from pathlib import Path
from services.admin.bechmarking import load_json_files, generate_qa_pairs, load_qa, group_sources
from dotenv import dotenv_values, set_key, load_dotenv
from config import Config
import os
from services.admin.buiding_augmenting_knowledge_base import build_initial_knowledge_base, add_new_disease

ENV_FILE = ".env"
qa_folder = f'{Config.EVAL}/generated_qa'
raw_json_files_folder = f'{Config.KB}/raw_files'


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
        json_files = load_json_files(raw_json_files_folder)
        print(f"Found {len(json_files)} JSON files")

        # Process each file
        for file_data in json_files:
            file_path = file_data['file_path']
            content = file_data['content']

            # Generate Q&A pairs
            qa_pairs = generate_qa_pairs(content)

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
    print(all_questions)
    print(all_answers)
    sum_score = 0
    # for i, question in enumerate(all_questions):
    #     sub_graphs, descriptions = retrieval(question, 1)
    #     answer = generate_response(descriptions, question, True)
    #
    #     response = generate_response(question)
    #     sub_graphs, descriptions = retreival(question, k=3)
    #     scores = evaluate_response(question, descriptions, response, all_answers[i], True)
    #     sum_score += float(scores["overall_feedback"])
    #     # print(sum_score)
    # average = sum_score / len(all_questions)
    # print(average)
    return {"hi": 1}
