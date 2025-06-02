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

"""
Code Report: admin_routes.py

1. Overview:
   This file implements a Flask Blueprint for admin-related routes in the application.
   It provides endpoints for managing settings, uploading data, and handling evaluations.

2. Dependencies:
   - Flask (Blueprint, jsonify, request)
   - Custom admin services from services.admin.admin_services

3. Routes:
   a) POST /upload
      - Purpose: Upload new text files and generate embeddings
      - Input: JSON with "path" field
      - Returns: Success message with 200 status code

   b) GET /settings
      - Purpose: Retrieve current application settings
      - Returns: Settings object with 200 status code

   c) POST /settings
      - Purpose: Update application settings
      - Input: JSON with settings data
      - Returns: Updated settings with 200 status code

   d) GET /dashboard
      - Purpose: Get knowledge base overview (currently unimplemented)
      - Status: Placeholder route, implementation pending

   e) POST /evaluate
      - Purpose: Trigger evaluation process
      - Input: JSON with "generate" boolean flag
      - Returns: Evaluation results with 200 status code

   f) GET /evaluation-report
      - Purpose: Retrieve evaluation report (currently unimplemented)
      - Status: Placeholder route, implementation pending

4. Code Structure:
   - Uses Flask Blueprint for route organization
   - Implements RESTful API endpoints
   - Follows consistent response pattern using jsonify
   - All routes return 200 status code on success

5. Areas for Improvement:
   - Implement missing dashboard functionality
   - Add evaluation report generation
   - Consider adding error handling
   - Add input validation for request data
   - Consider adding authentication/authorization
   - Add proper documentation for request/response formats

6. Security Considerations:
   - No visible authentication mechanism
   - Input validation could be enhanced
   - Consider adding rate limiting
   - Add proper error handling for file operations

7. Performance Considerations:
   - File upload operations might need optimization
   - Consider adding caching for settings
   - Evaluation process might need async handling

8. Testing Status:
   - No visible test coverage
   - Consider adding unit tests for each route
   - Add integration tests for file operations
"""
