from flask import Blueprint, request, jsonify
from db import db
from models import User, Conversation, Message, Response
from services.graph_embedding import graph_embedding_store
from services.retrieval import retrieval

user_bp = Blueprint('user_bp', __name__)


@user_bp.route("/ask", methods=["POST"])
def ask_question():
    # graph_embedding_store
    data = request.json
    user_id = data.get("user_id")
    question = data.get("question")

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Get or create conversation
    conversation = Conversation.query.filter_by(user_id=user.id).first()
    if not conversation:
        conversation = Conversation(user_id=user.id)
        db.session.add(conversation)
        db.session.commit()

    # Save message
    message = Message(conversation_id=conversation.id, text=question)
    db.session.add(message)

    # Run pipeline
    sub_graphs, descriptions = retrieval(question, 1)
    print(sub_graphs)
    print(descriptions)
    answer = "hi"
    # Save response
    response = Response(conversation_id=conversation.id, question=message, generated_answer=answer)
    db.session.add(response)
    db.session.commit()

    return jsonify({"answer": answer})
