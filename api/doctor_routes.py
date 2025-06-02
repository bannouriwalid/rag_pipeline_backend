from flask import Blueprint, request, jsonify
from models import ValidatedResponse, Response, Message
from db import db

doctor_bp = Blueprint("doctor_bp", __name__)


@doctor_bp.route("/inbox", methods=["GET"])
def get_pending_responses():
    pending = Response.query.filter_by(status="pending").all()

    results = []
    for resp in pending:
        message = Message.query.filter_by(id=resp.question_id).first()
        if message:
            results.append({
                "response_id": resp.id,
                "question": message.text,
                "answer": resp.generated_answer,
                "created_at": resp.created_at,
                "conversation_id": resp.conversation_id
            })

    return jsonify(results), 200


@doctor_bp.route("/validate/<int:response_id>", methods=["POST"])
def validate_response(response_id):
    data = request.json
    confirmed = data.get("confirmed")
    correction = data.get("correction")

    response = Response.query.get(response_id)
    if not response:
        return jsonify({"error": "Response not found"}), 404
    if response.status != "pending":
        return jsonify({"error": "Response already validated"}), 400

    if confirmed not in [True, False]:
        return jsonify({"error": "'confirmed' must be true or false"}), 400
    if not confirmed and not correction:
        return jsonify({"error": "Correction text required when not confirmed"}), 400

    # Save validated response
    validated = ValidatedResponse(
        response_id=response.id,
        confirmed=confirmed,
        correction=correction if not confirmed or correction else None
    )
    db.session.add(validated)

    # Update status & optionally add doctor's corrected answer
    response.status = "validated"
    db.session.commit()

    # Optionally notify user
    # (this could be email, push notification, or flag)
    # user_id = response.message.conversation.user_id
    # implement notify_user(user_id, message)

    return jsonify({"message": "Response validated", "response_id": response.id})

"""
Code Report: Doctor Routes Module
================================

Overview:
---------
This module implements a Flask Blueprint for doctor-specific API endpoints, handling the validation
and management of AI-generated medical responses.

Dependencies:
------------
- Flask: Blueprint, request, jsonify
- Local Models: ValidatedResponse, Response, Message
- Database: SQLAlchemy (db)

Endpoints:
----------
1. GET /inbox
   - Purpose: Retrieves all pending responses awaiting doctor validation
   - Returns: List of pending responses with:
     * response_id
     * question text
     * generated answer
     * creation timestamp
     * conversation_id
   - Status Codes: 200 (success)

2. POST /validate/<response_id>
   - Purpose: Validates or corrects an AI-generated response
   - Parameters:
     * response_id (path parameter)
     * confirmed (boolean)
     * correction (string, optional)
   - Validation Rules:
     * confirmed must be true or false
     * correction required if confirmed is false
   - Status Codes:
     * 200: Successful validation
     * 400: Invalid input or already validated
     * 404: Response not found

Database Models Used:
-------------------
1. Response
   - Fields: id, status, generated_answer, created_at, conversation_id
   - Status Values: "pending", "validated"

2. ValidatedResponse
   - Fields: response_id, confirmed, correction
   - Stores doctor's validation decision and corrections

3. Message
   - Fields: id, text, conversation_id
   - Stores the original question text

Security Considerations:
----------------------
- No authentication middleware implemented
- Consider adding:
  * Doctor authentication
  * Role-based access control
  * Input sanitization
  * Rate limiting

Future Improvements:
-------------------
1. Implement user notification system
2. Add pagination for inbox endpoint
3. Add filtering options for responses
4. Implement audit logging
5. Add response validation metrics
6. Implement caching for frequently accessed data

Error Handling:
--------------
- Basic error handling implemented
- Returns appropriate HTTP status codes
- Provides descriptive error messages

Performance Considerations:
-------------------------
- Uses SQLAlchemy ORM for database operations
- Consider adding:
  * Database indexing on frequently queried fields
  * Response caching
  * Query optimization for large datasets
"""
