import re
from config import Config
from openai import OpenAI
from models import Evaluation
from db import db


def evaluate_response(question, context, response, label=None):
    evaluation_prompt = f"""You are an expert evaluator of medical knowledge responses. Evaluate the following response based on three criteria:

1. Clarity (0-5): How clear and well-structured is the response? 0 is the worst, 5 is the best.
2. Exactitude (0-5): How accurate and precise is the information provided? 0 is the worst, 5 is the best.
3. Context Adherence (0-5): How well does the response stick to the provided knowledge graphs? 0 is the worst, 5 is the best.
4. Relevance (0-5): How relevant is the retrieved Knowledge Graph Context to the question? 0 is the worst, 5 is the best.
5. Completeness (0-5): How complete and thorough is the response? 0 is the worst, 5 is the best.
6. Logical Flow (0-5): How coherent and well-structured is the response? 0 is the worst, 5 is the best.
7. Uncertainty Handling (0-5): How well does the response acknowledge limitations and uncertainties? 0 is the worst, 5 is the best.


Question: {question}

Knowledge Graph Context:
{context}

Response to Evaluate:
{response}

Provide your evaluation in the following format:
CLARITY: [score]/5 - [brief explanation]
EXACTITUDE: [score]/5 - [brief explanation]
CONTEXT ADHERENCE: [score]/5 - [brief explanation]
RELEVANCE: [score]/5 - [brief explanation]
COMPLETENESS: [score]/5 - [brief explanation]
LOGICAL FLOW: [score]/5 - [brief explanation]
UNCERTAINTY HANDLING: [score]/5 - [brief explanation]
OVERALL FEEDBACK: [average score] and 2-3 sentences summarizing the evaluation]
"""

    if label is not None:
        evaluation_prompt += f"Ground Truth Label: {label}\n"

    # new client (llm api) for evaluation different  meta/llama-3.1-405b-instruct
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=Config.NVIDIA_API_KEY
    )
    evaluation = client.chat.completions.create(
        model=Config.EVALUATION_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert evaluator of medical knowledge responses."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2,  # Lower temperature for more consistent evaluation
        top_p=0.95,
        max_tokens=1024,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )

    # Initialize variables to store the complete response
    full_evaluation = ""
    scores = {
        "clarity_score": None,
        "exactitude_score": None,
        "context_adherence_score": None,
        "relevance_score": None,
        "completeness_score": None,
        "logical_flow_score": None,
        "uncertainty_handling_score": None,
        "overall_feedback": None
    }

    for chunk in evaluation:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_evaluation += content

    # Extract scores from the complete evaluation
    scores["clarity_score"] = re.findall(r"(?i)clarity\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["exactitude_score"] = re.findall(r"(?i)EXACTITUDE\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["context_adherence_score"] = re.findall(r"(?i)CONTEXT ADHERENCE\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["relevance_score"] = re.findall(r"(?i)RELEVANCE\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["completeness_score"] = re.findall(r"(?i)COMPLETENESS\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["logical_flow_score"] = re.findall(r"(?i)LOGICAL FLOW\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["uncertainty_handling_score"] = re.findall(r"(?i)UNCERTAINTY HANDLING\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)
    scores["overall_feedback"] = re.findall(r"(?i)OVERALL FEEDBACK\s*:\s*(\d+(?:\.\d+)?)", full_evaluation)

    # if the response is "I don't know" set the scores
    if response.strip().lower().startswith("i can't answer this question") or response.strip().lower().startswith("i don't know"):
        scores["clarity_score"] = 0
        scores["exactitude_score"] = 0
        scores["context_adherence_score"] = 0
        scores["relevance_score"] = 0
        scores["completeness_score"] = 0
        scores["logical_flow_score"] = 0
        scores["uncertainty_handling_score"] = 0
        scores["overall_feedback"] = 0

    # Convert list matches to single values
    for key in scores:
        try:
            if scores[key]:
                if isinstance(scores[key], list):
                    scores[key] = float(scores[key][0])
                else:
                    scores[key] = float(scores[key])
        except ValueError:
            scores[key] = None

    evaluation = Evaluation(question=question, context=context,
                            response=response, label=label,
                            clarity_score=scores["clarity_score"],
                            exactitude_score=scores["exactitude_score"],
                            context_adherence_score=scores["context_adherence_score"],
                            relevance_score=scores["relevance_score"],
                            completeness_score=scores["completeness_score"],
                            logical_flow_score=scores["logical_flow_score"],
                            uncertainty_handling_score=scores["uncertainty_handling_score"],
                            overall_feedback=scores["overall_feedback"])
    db.session.add(evaluation)
    db.session.commit()

"""
Code Report: Evaluation Module
=============================

Overview:
---------
This module implements an automated evaluation system for medical knowledge responses using NVIDIA's LLM API.
The system evaluates responses based on multiple criteria and stores the results in a database.

Key Components:
--------------
1. Main Function: evaluate_response(question, context, response, label=None)
   - Takes question, context, response, and optional label as inputs
   - Returns evaluation scores and feedback

2. Evaluation Criteria (0-5 scale):
   - Clarity: Response structure and presentation
   - Exactitude: Accuracy and precision
   - Context Adherence: Alignment with knowledge graphs
   - Relevance: Context relevance to question
   - Completeness: Thoroughness of response
   - Logical Flow: Coherence and structure
   - Uncertainty Handling: Acknowledgment of limitations

3. Technical Implementation:
   - Uses NVIDIA's LLM API (meta/llama-3.1-405b-instruct)
   - Implements streaming response handling
   - Uses regex for score extraction
   - Includes special handling for "I don't know" responses

4. Database Integration:
   - Stores evaluations in a database using SQLAlchemy
   - Creates Evaluation objects with all scores and metadata

Security & Configuration:
------------------------
- Uses Config class for API keys and model settings
- Implements secure API key handling
- Uses environment variables for sensitive data

Performance Considerations:
-------------------------
- Uses streaming for large responses
- Implements efficient regex pattern matching
- Handles response chunks incrementally

Error Handling:
--------------
- Gracefully handles missing or invalid scores
- Converts string scores to float values
- Handles special cases (e.g., "I don't know" responses)

Dependencies:
------------
- openai: For LLM API integration
- re: For regex pattern matching
- SQLAlchemy: For database operations
- Custom modules: Config, Evaluation model, db

Maintenance Notes:
----------------
- Regular updates needed for evaluation criteria
- API key rotation should be implemented
- Consider adding logging for debugging
- May need periodic review of scoring patterns

Future Improvements:
-------------------
1. Add input validation
2. Implement retry mechanism for API calls
3. Add comprehensive error logging
4. Consider caching frequently used evaluations
5. Add unit tests for score extraction
6. Implement batch processing capability
"""
