"""
Sequential Q&A processor for climsight questions.

This script reads a JSON file containing questions organized by themes and categories.
For each question, it uses a Climsight to generate an answer based on the provided location.
Answers are added back into the original data structure and saved to an output JSON file.

Additionally, each answer is appended to a temporal JSONL file during processing,
allowing progress to be preserved in case of interruption.

Arguments such as input file, config path, model, and output paths are provided via command-line flags.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

# Store script directory for relative paths
script_dir = Path(__file__).parent

# Add project root to sys.path and change working directory
project_root = script_dir.parent
os.chdir(project_root)  # Change working directory to project root
sys.path.insert(0, str(project_root / 'src' / 'climsight'))

from terminal_interface import run_terminal




# Set up paths - now relative to project root
project_root = Path('.')

# Create the parser
parser = argparse.ArgumentParser(description="Process the arguments")
# Add the argument with a default value of an empty string
parser.add_argument('--questions_file', type=str, 
                    default=str(script_dir / 'questions.json'), 
                    help='Path to the QA file')
parser.add_argument('--config_path', type=str, 
                    default=str(project_root / 'config.yml'), 
                    help='Path to the config file')
parser.add_argument('--api_key', type=str, 
                    default='', 
                    help='API key for the OpenAI API')
parser.add_argument('--llm_model', type=str, 
                    default='gpt-4o', 
                    help='Model used for the Climsight answer generation (default: gpt-4o)')
parser.add_argument('--file_answers', type=str, 
                    default=str(script_dir / 'answers.yml'), 
                    help='Path to the file with answers from Climsight')
parser.add_argument('--output_file', type=str, 
                    default=str(script_dir / 'answers_output.json'), 
                    help='Path to the file with answers from Climsight')
parser.add_argument('--temporal_file', type=str, 
                    default=str(script_dir / 'temporal_output.json'), 
                    help='Path to the temporal file with answers from Climsight')


# Parse the arguments
args = parser.parse_args()

# Ensure output directory exists - use absolute paths for file operations
output_file = project_root / args.output_file
temporal_file = project_root / args.temporal_file
output_dir = output_file.parent
output_dir.mkdir(parents=True, exist_ok=True)
temporal_dir = temporal_file.parent
temporal_dir.mkdir(parents=True, exist_ok=True)
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
# Load the config file
config_path = (project_root / args.config_path).resolve()
print(f"Loading config from: {config_path}")
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading config file {config_path}: {e}")
    sys.exit(1)



# Resolve question file path relative to the script location
questions_file_path = (script_dir / args.questions_file).resolve()
print(f"Loading questions from: {questions_file_path}")
try:
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
except Exception as e:
    print(f"Error loading questions file {questions_file_path}: {e}")
    sys.exit(1)



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
question_count = 0
for theme, categories in questions['themes'].items():
    for category_type in ('general', 'specific'):
        for q in categories.get(category_type, []):
            question_count += 1
            print(f"!!!!!!!!!!!!!!!!!!!!!!!! Processing question {question_count} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(theme, category_type, q['question'], q['lon'], q['lat'])
            try:
                user_message = q['question']
                lon = q.get('lon', 13.37)
                lat = q.get('lat', 52.524)
                output, input_params, content_message, combine_agent_prompt_text  = run_terminal(config, 
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
                combine_agent_prompt_text = ""

            q['answer'] = output  
            q['combine_agent_prompt_text'] = combine_agent_prompt_text              
            # Add context to the output
            answer_entry = {
                "theme": theme,
                "category": category_type,
                "question": user_message,
                "lat": lat,
                "lon": lon,
                "answer": output,
                "combine_agent_prompt_text": combine_agent_prompt_text
            }

            output_answers.append(answer_entry)

            # Dump answer inside loop at each iteration
            with open(project_root / temporal_file, 'a', encoding='utf-8') as f:
                json.dump(answer_entry, f, ensure_ascii=False)
                f.write('\n')  # Each entry on a new line     
                
                
# Dump the updated structure after each question
with open(project_root / output_file, 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)       

