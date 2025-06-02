import json
import glob
import os
from config import Config
from openai import OpenAI


def group_sources(path):
    only_dirs = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

    for folder in only_dirs:
        all_data = []

        for filename in os.listdir(os.path.join(path, folder)):
            try:
                with open(os.path.join(path, folder)+'/'+filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

        output_file = path+f'/all_{folder}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2)


def load_qa(qa_folder):
    # Get all QA JSON files from the generated_qa directory
    qa_files = glob.glob(qa_folder + '/*.json')

    # Initialize lists to store all questions and answers
    all_questions = []
    all_answers = []

    # Load data from each file
    for file_path in qa_files:
        try:
            # Load JSON data from the file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if the loaded data is a list
            if isinstance(data, list):
                # Iterate through the list of Q&A pairs
                for qa_pair in data:
                    # Ensure each item in the list is a dictionary with 'question' and 'answer' keys
                    if isinstance(qa_pair, dict) and 'question' in qa_pair and 'answer' in qa_pair:
                        all_questions.append(qa_pair['question'])
                        all_answers.append(qa_pair['answer'])
                    else:
                        print(f"Warning: Skipping invalid item in {file_path}: {qa_pair}")
            else:
                print(f"Warning: Data in {file_path} is not a list. Skipping file.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # return questions and answers
    return all_questions, all_answers


def load_json_files(base_dir):
    json_files = []
    for json_file in glob.glob(os.path.join(base_dir, "*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_files.append({
                    'file_path': json_file,
                    'content': data
                })
            print(f"Successfully loaded {json_file}")
        except Exception as error:
            print(f"Error loading {json_file}: {str(error)}")
    return json_files


def generate_qa_pairs(content):
    # System prompt to set the context and behavior
    system_prompt = """You are a medical expert tasked with generating high-quality question-answer pairs from medical information.
    Your responses should be:
    1. Medically accurate and evidence-based
    2. Clear and concise
    3. Appropriate for both medical professionals and patients
    4. Focused on key medical concepts, treatments, and implications
    5. Free from any harmful or misleading information

    Format your response as a JSON array of objects with 'question' and 'answer' fields."""

    # User prompt to guide the specific generation task
    user_prompt = f"""Based on the following medical information, generate relevant question-answer pairs.
    Make the questions specific and the answers detailed but concise.

    Information:
    {json.dumps(content, indent=2)}

    Return ONLY a JSON array of objects with 'question' and 'answer' fields, with no additional text or markdown formatting.
    Example format:
    [
        {{
            "question": "What causes tuberculosis?",
            "answer": "Tuberculosis is caused by the bacterium Mycobacterium tuberculosis."
        }}
    ]"""

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=Config.NVIDIA_API_KEY
        )
        # Make the API call with both prompts
        stream = client.chat.completions.create(
            model=Config.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            top_p=0.95,
            max_tokens=2048,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if "choices" in chunk and len(chunk.choices) > 0:
                full_response += chunk.choices[0].delta.get("content", "")

        # Remove possible markdown formatting
        if "```json" in full_response:
            full_response = full_response.split("```json")[1]
        if "```" in full_response:
            full_response = full_response.split("```")[0]

        full_response = full_response.strip()

        # Parse the response as JSON
        try:
            qa_pairs = json.loads(full_response)
            if not isinstance(qa_pairs, list):
                raise ValueError("Response is not a list")
            for pair in qa_pairs:
                if not isinstance(pair, dict) or 'question' not in pair or 'answer' not in pair:
                    raise ValueError("Invalid Q&A pair format")

            print("qa_pairs")
            print(qa_pairs)
            return qa_pairs
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            return []

    except Exception as e:
        print(f"Error generating Q&A pairs: {str(e)}")
        return []
