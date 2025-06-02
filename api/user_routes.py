from flask import Blueprint, request, jsonify
from db import db
from models import User, Conversation, Message, Response, ValidatedResponse
from services.user.response_generation import generate_response
from services.user.retrieval import retrieval

user_bp = Blueprint('user_bp', __name__)


@user_bp.route("/ask", methods=["POST"])
def ask_question():
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

    # Run RAG pipeline
    sub_graphs, descriptions = retrieval(question, 1)
    answer = generate_response(descriptions, question, True)

    # Save response
    response = Response(conversation_id=conversation.id, question_id=message.id, generated_answer=answer)
    db.session.add(response)
    db.session.commit()

    return jsonify({"answer": answer})


@user_bp.route("/conversation", methods=["GET"])
def get_conversation_history():
    user_id = request.args.get("user_id", type=int)

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    conversation = Conversation.query.filter_by(user_id=user.id).first()
    if not conversation:
        return jsonify([]), 200

    # Get all messages and responses
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at).all()

    result = []
    for msg in messages:
        resp = Response.query.filter_by(question_id=msg.id).first()
        result.append({
            "question_created_at": msg.created_at,
            "question": msg.text,
            "answer_created_at": resp.created_at if resp else None,
            "answer": resp.generated_answer if resp else None,
            "status": resp.status if resp else None,
        })

    return jsonify(result)


@user_bp.route("/inbox", methods=["GET"])
def get_validated_responses():
    user_id = request.args.get("user_id", type=int)
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    conversation = Conversation.query.filter_by(user_id=user.id).first()
    if not conversation:
        return jsonify([])
    responses = Response.query.filter_by(conversation_id=conversation.id, status="validated").all()
    validated_responses = [
        ValidatedResponse.query.filter_by(response_id=r.id).first()
        for r in responses
    ]

    results = []

    for i in range(len(validated_responses)):
        message = Message.query.filter_by(id=responses[i].question_id).first()
        if message:
            results.append({
                "question": message.text,
                "question_created_at": message.created_at,
                "llm_answer": responses[i].generated_answer,
                "answer_created_at": responses[i].created_at,
                "status": responses[i].status,
                "confirmed": validated_responses[i].confirmed,
                "correction": validated_responses[i].correction,
                "validation_created_at": validated_responses[i].created_at
            })
    return jsonify(results)

"""
Code Report: api/user_routes.py

1. Overview
-----------
This file implements a Flask Blueprint for user-related API endpoints in a RAG (Retrieval-Augmented Generation) pipeline backend.
The module handles user interactions, conversation management, and response generation/validation.

2. Dependencies
--------------
- Flask: Web framework
- Database models: User, Conversation, Message, Response, ValidatedResponse
- Services: response_generation, retrieval

3. API Endpoints
---------------
a) POST /ask
   - Purpose: Handles user questions and generates AI responses
   - Input: JSON with user_id and question
   - Process:
     * Validates user existence
     * Creates/retrieves conversation
     * Saves user message
     * Runs RAG pipeline (retrieval + response generation)
     * Saves generated response
   - Output: JSON with generated answer

b) GET /conversation
   - Purpose: Retrieves conversation history
   - Input: user_id as query parameter
   - Process:
     * Validates user existence
     * Retrieves all messages and responses
     * Orders by creation time
   - Output: JSON array of Q&A pairs with timestamps

c) GET /inbox
   - Purpose: Retrieves validated responses
   - Input: user_id as query parameter
   - Process:
     * Validates user existence
     * Retrieves validated responses
     * Includes original questions and validation details
   - Output: JSON array of validated Q&A pairs

4. Data Flow
-----------
- User input → Message storage → RAG processing → Response storage
- Conversation tracking across multiple interactions
- Validation status tracking for responses

5. Error Handling
----------------
- User not found: Returns 404 status
- Empty conversations: Returns empty arrays
- Basic input validation for required fields

6. Database Integration
----------------------
- Uses SQLAlchemy ORM
- Maintains relationships between:
  * Users and Conversations
  * Conversations and Messages
  * Messages and Responses
  * Responses and ValidatedResponses

7. Security Considerations
-------------------------
- User authentication required (user_id validation)
- Input validation for required fields
- No direct exposure of internal data structures

8. Performance Considerations
---------------------------
- Efficient database queries with proper indexing
- Ordered retrieval of conversation history
- Batch processing of validated responses

9. Future Improvements
---------------------
- Add pagination for conversation history
- Implement rate limiting
- Add more robust error handling
- Consider caching frequently accessed data
- Add input sanitization
- Implement proper authentication middleware
"""
