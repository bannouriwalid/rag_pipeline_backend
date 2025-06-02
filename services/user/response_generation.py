from openai import OpenAI
from config import Config
import re
from services.user.evaluation import evaluate_response


def construct_textualize_prompt_messages(description):
    # Start with a system message or instruction
    system_message = """You are a helpful assistant that summarizes medical knowledge graphs into coherent text.
Based on the provided knowledge graph nodes and relationships, generate a concise and informative paragraph describing the medical concepts and their connections.
Do not include node IDs or graph numbers in your summary. Be very concise and stick to the sub-graphs and descriptions provided. Don't add any new information
"""

    # Format the context from the knowledge graph description
    # Split the description into nodes and edges
    nodes = description[0]
    edges = description[1] if len(description) > 1 else ""

    context = "Here is a medical knowledge graph description:\n"
    context += f"Nodes:\n{nodes}\n"
    if edges:
        context += f"Relationships:\n{edges}\n"

    # Construct the final prompt
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context}
    ]


def textualize(descriptions):
    # Construct the prompt for the LLM
    prompt_messages = construct_textualize_prompt_messages(descriptions)

    # Initialize OpenAI or NVIDIA client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=Config.NVIDIA_API_KEY
    )

    try:
        # Make the API call to the LLM
        chat_completion = client.chat.completions.create(
            messages=prompt_messages,
            model=Config.TEXTUALIZATION_MODEL
        )

        # Extract the response
        llm_response = chat_completion.choices[0].message.content
        return llm_response

    except Exception as e:
        print(f"Error during LLM API call: {e}")
        return "An error occurred while generating the response."


def construct_generation_prompt_messages(question, context):
    system_message = """You are a helpful and knowledgeable medical assistant specialized in respiratory illnesses.
You assist patients by answering questions clearly and accurately using only the provided medical context.
Avoid technical jargon where possible and keep your answers friendly and easy to understand.
Only use the information in the context. If the context is not sufficient to answer the question, say "I don't know."
If the context is unrelated, say "I can't answer this question." Do not mention the source of the information in your response.
Be concise, natural, and supportive in your tone.
"""

    # Format the prompt
    full_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": full_prompt}
    ]


def generate_response(descriptions, question, is_evaluated=False):
    try:
        context = textualize(descriptions)
        rag_prompt_messages = construct_generation_prompt_messages(question, context)

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=Config.NVIDIA_API_KEY
        )

        completion = client.chat.completions.create(
            model=Config.GENERATION_MODEL,
            messages=rag_prompt_messages,
            temperature=0.4,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )

        #   words to delete from the response:
        to_delete = [
            r"\(Knowledge Graph \d+\)",
            r"Knowledge Graph \d+"
        ]

        # Print the streaming response and save the response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                for word in to_delete:
                    content = re.sub(word, "", content)
                full_response += content

    except Exception as e:
        print(f"Error during response generation: {e}")
        return "An error occurred while generating the final answer."

    if is_evaluated:
        try:
            evaluate_response(question, descriptions, full_response)
        except Exception as e:
            print(f"An error occurred while evaluating the answer: {e}")

    return full_response


"""
Code Report: Response Generation Module
=====================================

Overview:
---------
This module implements a medical knowledge graph-based response generation system using NVIDIA's API.
It provides functionality to convert knowledge graph data into natural language responses for medical queries.

Key Components:
--------------
1. textualize(descriptions)
   - Converts knowledge graph nodes and relationships into coherent text
   - Uses NVIDIA's API for text generation
   - Handles error cases gracefully

2. generate_response(descriptions, question, is_evaluated=False)
   - Main function for generating responses to medical queries
   - Combines knowledge graph textualization with question answering
   - Supports optional response evaluation
   - Implements streaming response generation
   - Includes post-processing to remove unwanted patterns

3. Helper Functions:
   - construct_textualize_prompt_messages(): Prepares prompts for knowledge graph textualization
   - construct_generation_prompt_messages(): Prepares prompts for response generation

Technical Details:
----------------
- Uses OpenAI client with NVIDIA API integration
- Implements streaming responses for better user experience
- Includes error handling and logging
- Uses regex for response cleaning
- Configurable through Config class

Dependencies:
------------
- openai: For API client implementation
- config: For configuration management
- re: For regular expression operations
- services.user.evaluation: For response evaluation

Security & Performance:
----------------------
- API keys managed through Config class
- Streaming implementation for efficient response generation
- Error handling for API failures
- Response cleaning for consistent output

Areas for Improvement:
---------------------
1. Add retry mechanism for API calls
2. Implement caching for frequently asked questions
3. Add more comprehensive logging
4. Consider implementing rate limiting
5. Add input validation for descriptions and questions

Usage Example:
-------------
```python
descriptions = [nodes, edges]
question = "What are the symptoms of asthma?"
response = generate_response(descriptions, question, is_evaluated=True)
```

Note: This module is specifically designed for medical knowledge graph processing and should be used with appropriate medical context validation.
"""

