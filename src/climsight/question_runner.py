"""
Sequential Q&A processor for climsight questions.

This script reads a JSON file containing questions organized by themes and categories.
For each question, it uses a Climsight to generate an answer based on the provided location.
Answers are added back into the original data structure and saved to an output JSON file.

Additionally, each answer is appended to a temporal JSONL file during processing,
allowing progress to be preserved in case of interruption.

Arguments such as input file, config path, model, and output paths are provided via command-line flags.
"""

import sys
import os

# Add src/climsight to sys.path
sys.path.insert(0, os.path.abspath("src/climsight"))

from terminal_interface import run_terminal
import json
import yaml

import os
import sys
import argparse



# Create the parser
parser = argparse.ArgumentParser(description="Process the arguments")
# Add the argument with a default value of an empty string
parser.add_argument('--questions_file', type=str, default='sequential/questions.json', help='Path to the QA file')
parser.add_argument('--config_path', type=str, default='config.yml', help='Path to the config file file')
parser.add_argument('--api_key', type=str, default='', help='API key for the OpenAI API')
#parser.add_argument('--api_key_local', type=str, default='', help='API key for the local API LLM')
parser.add_argument('--llm_model', type=str, default='gpt-4o', help='model used for the Climsight answer generation (default: gpt-4o)')
parser.add_argument('--file_answers', type=str, default='sequential/answers.yml', help='Path to the file with answers from Climsight')
parser.add_argument('--output_file', type=str, default='sequential/answers_output.json', help='Path to the file with answers from Climsight')
parser.add_argument('--temporal_file', type=str, default='sequential/temporal_output.jsonl', help='Path to the temporal file with answers from Climsight')


# Parse the arguments
args = parser.parse_args()

#questions_file = os.path.join("sequential","questions.json")
#config_path = os.path.join("config.yml")
#llm_model = 'gpt-4o'
#file_answers = os.path.join("sequential", "answers_climsight.yml")
#api_key = ''
#output_file = os.path.join("sequential", "answers_output.json")
#temporal_file= os.path.join("sequential", "temporal_output.jsonl")

questions_file = args.questions_file
config_path = args.config_path
api_key = args.api_key
#api_key_local = args.api_key_local
llm_model = args.llm_model
file_answers = args.file_answers
output_file = args.output_file
temporal_file = args.temporal_file


if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY") # get the API key from the environment variable
    if not api_key:
        raise ValueError("API key is not provided in the arguments or in the environment variable")

#if not api_key_local:
#    api_key_local = os.environ.get("OPENAI_API_KEY_LOCAL")
#    if not api_key_local:
#        api_key_local = ""
    
print(f"reading config from: {config_path}")
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"An ERROR occurred while reading the file: {config_path}")
    raise RuntimeError(f"An error occurred while reading the file: {config_path}") from e



print(f"reading Questions from: {questions_file}")
# Load the JSON file
try:
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
except Exception as e:
    print(f"An ERROR occurred while reading the file: {questions_file}")
    raise RuntimeError(f"An ERROR occurred while reading the file: {questions_file}") from e



references = None

rag_settings = config['rag_settings']
embedding_model = rag_settings['embedding_model']
chroma_path_ipcc = rag_settings['chroma_path_ipcc']
chroma_path_general = rag_settings['chroma_path_general'] 
chroma_path = [chroma_path_ipcc, chroma_path_general]

#set LLM to be used for the Climsight 
config['model_name_combine_agent'] = llm_model

output_answers = []

# If the file exists, load existing data
#if os.path.exists(output_file):
#    with open(output_file, 'r', encoding='utf-8') as f:
#        try:
#            output_answers = json.load(f)
#        except json.JSONDecodeError:
#            output_answers = []
            
# Loop over all themes
for theme, categories in questions['themes'].items():
    for category_type in ('general', 'specific'):
        for q in categories.get(category_type, []):
            
            print(theme, category_type, q['question'], q['lon'], q['lat'])
            try:
                user_message = q['question']
                lon = q.get('lon', 13.37)
                lat = q.get('lat', 52.524)
                output  = run_terminal(config, 
                                    skip_llm_call=False, 
                                    lon=lon, lat=lat, 
                                    user_message=user_message, 
                                    show_add_info='n', 
                                    verbose=False, 
                                    rag_activated=True, 
                                    embedding_model=embedding_model, 
                                    chroma_path=chroma_path,
                                    references=references)    

            except Exception as e:
                print(f"An ERROR occurred while processing the question: {q['question']}")
                print(e)
                output = None

            q['answer'] = output                
            # Add context to the output
            answer_entry = {
                "theme": theme,
                "category": category_type,
                "question": user_message,
                "lat": lat,
                "lon": lon,
                "answer": output
            }

            output_answers.append(answer_entry)

            # Dump answer inside loop at each iteration
            with open(temporal_file, 'a', encoding='utf-8') as f:
                json.dump(answer_entry, f, ensure_ascii=False)
                f.write('\n')  # Each entry on a new line     
                
                
# Dump the updated structure after each question
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)       

