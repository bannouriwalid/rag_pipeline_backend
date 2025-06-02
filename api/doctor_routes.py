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
