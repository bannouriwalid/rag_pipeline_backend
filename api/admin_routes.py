from flask import Blueprint, jsonify, request
from services.admin.admin_services import (
    update_settings,
    get_settings,
    add_new_text_file,
    trigger_evaluation
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


@admin_bp.route("/dashboard", methods=["GET"])
def dashboard():
    pass
    # overview = get_kb_overview()
    # return jsonify(overview), 200


@admin_bp.route("/evaluate", methods=["POST"])
def run_evaluation():
    data = request.json
    generate_qa = data.get("generate")
    result = trigger_evaluation(generate_qa)
    return jsonify(result), 200


@admin_bp.route("/evaluation-report", methods=["GET"])
def evaluation_report():
    pass
    # report = get_evaluation_report()
    # return jsonify(report), 200
