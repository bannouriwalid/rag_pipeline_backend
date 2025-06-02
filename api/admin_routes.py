import os
from flask import Blueprint, jsonify, request
from config import Config
from services.admin.admin_services import (
    update_settings,
    get_settings,
    add_new_text_file,
    trigger_evaluation, get_kb_overview, get_overview
)

admin_bp = Blueprint('admin_bp', __name__)


@admin_bp.route("/upload", methods=["POST"])
def upload_data():
    data = request.json
    path = data.get("path")
    add_new_text_file(path)
    return jsonify({"message": "Graphs uploaded and embeddings generated successfully."}), 200


@admin_bp.route("/settings", methods=["GET"])
def view_settings():
    settings = get_settings()
    return jsonify(settings), 200


@admin_bp.route("/settings", methods=["POST"])
def modify_settings():
    data = request.json
    result = update_settings(data)
    return jsonify(result), 200


@admin_bp.route("/evaluate", methods=["POST"])
def run_evaluation():
    data = request.json
    generate_qa = data.get("generate")
    result = trigger_evaluation(generate_qa)
    return jsonify(result), 200


@admin_bp.route("/evaluation-report", methods=["POST"])
def evaluation_report():
    EVALUATION_FOLDER = f'{Config.EVAL}'
    data = request.json
    file_name = data.get("file_name")

    # Sanitize and validate file name
    if not file_name or ".." in file_name or "/" in file_name or "\\" in file_name:
        return jsonify({"error": "Invalid file name"}), 400

    file_path = os.path.join(EVALUATION_FOLDER, file_name)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return jsonify({"content": content}), 200


@admin_bp.route("/dashboard", methods=["GET"])
def dashboard():
    general_overview = get_overview()
    kb_overview = get_kb_overview()
    combined_overview = {
        **general_overview,
        "knowledge_base": kb_overview
    }

    return jsonify(combined_overview), 200
