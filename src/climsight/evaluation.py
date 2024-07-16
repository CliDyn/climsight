from terminal_interface import run_terminal
import logging
import yaml
import os
import sys
import argparse
import re
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Initialize logging at the beginning of your main application
logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight_evaluation.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


# Create the parser
parser = argparse.ArgumentParser(description="Process the arguments")
# Add the argument with a default value of an empty string
parser.add_argument('--qa_file', type=str, default='', help='Path to the QA file')
parser.add_argument('--config_path', type=str, default='', help='Path to the config file file')
parser.add_argument('--api_key', type=str, default='', help='API key for the OpenAI API')
parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-0125', help='model used for the evaluation (default: gpt-3.5-turbo-0125)')
# Parse the arguments
args = parser.parse_args()

qa_file = args.qa_file
config_path = args.config_path
api_key = args.api_key

if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY") # get the API key from the environment variable
    if not api_key:
        raise ValueError("API key is not provided in the arguments or in the environment variable")
                         
# model used for the evaluation
#llm_model = 'gpt-3.5-turbo-0125'
#llm_model = 'gpt-4o'
llm_model = args.llm_model


# if config path is not provided in arguments, use ENV variable, otherwise use the default value
if not config_path:
    config_path = os.getenv('CONFIG_PATH', 'config.yml')
  
logger.info(f"reading config from: {config_path}")
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    logging.error(f"An error occurred while reading the file: {config_path}")
    raise RuntimeError(f"An error occurred while reading the file: {config_path}") from e

# if qa path is not provided in arguments, use default value
if not qa_file:
    qa_file = os.path.join("evaluation", "QA.yml")

logger.info(f"reading QA from: {qa_file}")
try:
    with open(qa_file, 'r') as file:
        question_answers = yaml.safe_load(file)
except Exception as e:
    logging.error(f"An error occurred while reading the file: {qa_file}")
    raise RuntimeError(f"An error occurred while reading the file: {qa_file}") from e

evaluation_template = """You are the evaluation expert. You are going to compare two answers. The first answer is the correct answer. The second answer is from the project Climsight. You will also have the question to which the answers were given.

Follow these steps:

	1.	Understand the Question Context:
	•	Read the provided question carefully.
	•	Identify the key elements and specifics of what the question is asking regarding the effects of climate change on human activities.
	2.	Analyze the Provided Answers:
	•	Read the provbided answer from Climsight to evaluate.
	•	Read the provided correct answer to compare with.
	•	Identify and note the main points, key arguments, and any supporting data or examples in both answers.
	3.	Compare Answers for Completeness:
	•	Evaluate if the Climsight answer covers all the key points mentioned in the correct answer.
	•	Check if there are any critical elements from the correct answer that are missing or inadequately covered in the Climsight answer.
	4.	Assess Accuracy and Relevance:
	•	Determine the accuracy of the information provided by Climsight by comparing it with the correct answer.
	5.	Evaluate Clarity and Coherence:
	•	Assess the clarity of the Climsight answer. Is it easy to understand and logically presented?
	•	Compare the coherence of both answers. Does Climsight`s answer follow a logical flow and structure similar to the correct answer?
	6.	Rate the Answer:
	•	Assign a score for completeness on a scale of 1 to 5, where 1 is the lowest and 5 is the highest.
	•	Assign a score for accuracy on a scale of 1 to 5, where 1 is the lowest and 5 is the highest.
	•	Assign a score for relevance on a scale of 1 to 5, where 1 is the lowest and 5 is the highest.
	•	Assign a score for clarity and coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest.
	7.	Provide Feedback and Suggestions:
	•	Summarize the strengths and weaknesses of the Climsight answer.
	•	Provide specific suggestions on how the Climsight answer can be improved to better match the correct answer.

Question: {question}
Answer 1 (correct one): {correct_answer}
Answer 2 (evaluate this answer, climsight): {climsight_answer}

You alkso need to provide a recomendation on how to improve the answer from Climsight, for example what kind of data are missing.

at the end of the evaluation, you will provide a table with scores following template below:
template for the table, replace the <your score> with the actual scores of your evaluation, do not change separators "," use only "," to separate the values in the table:

#### Scores Table:
Criteria, Score (1-5)
Completeness, <your score>
Accuracy, <your score>
Relevance, <your score>
Clarity and Coherence, <your score>
Mean, <your score>
"""
valid_criteria = ["Completeness", "Accuracy", "Relevance", "Clarity and coherence", "Mean"]

def request_answers_from_climsight(question_answers, series = 'ipcc'):
    """_summary_
    return answers from the climsight for the given series of questions

    Args:
        question_answers (dictionary): dictionary with questions, answers, promts, lon, lat ...
        series (str, optional): _description_. Defaults to 'ipcc'.

    Returns:
        output_answers (dictionry): Answers from the climsight
    """
    # Create a dictionary to store the outputs
    output_answers = []

    # Iterate over the questions and run the terminal command
    for qa_indx in range(len(question_answers[series])):
        user_message = question_answers[series][qa_indx].get('promt', '')
        user_message = user_message + ' ' + question_answers[series][qa_indx]['question']
        lon = question_answers[series][qa_indx].get('lon', 13.37)
        lat = question_answers[series][qa_indx].get('lat', 52.524)
        output = run_terminal(config, skip_llm_call=False, lon=lon, lat=lat, user_message=user_message, show_add_info='n', verbose=False)
        output_answers.append(output)

    return output_answers

def evaluate_answers(question_answers, climsight_answers, series, llm_model, evaluation_template):
    """_summary_
       Evaluate the answers from the Climsight by comparing them with the correct answers 
    Args:
        question_answers (dict): questions with the correct answers
        climsight_answers (_type_): answers on the question from question_answers with the Climsight
        series (_type_): 'ipcc' or 'gerics', or ...1
        llm_model (_type_): type of the model used for the evaluation gp-3.5-turbo-0125, gpt-4o, ...
        evaluation_template (_type_): template for the evaluation

    Returns:
        dict : return the evaluation of the answers
    """
    llm = ChatOpenAI(model=llm_model, api_key=api_key)
    custom_rag_prompt = PromptTemplate.from_template(evaluation_template)
 
    rag_chain = (
        {"correct_answer": RunnablePassthrough(), "climsight_answer": RunnablePassthrough(), "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    evaluation = []
    
    for qa_indx in range(len(climsight_answers)):
        print("Evaluation for question: ", qa_indx+1)
        # Prepare inputs for the chain
        question =  question_answers[series][qa_indx]['question']
        correct_answer = question_answers[series][qa_indx]['answer']
        climsight_answer = climsight_answers[qa_indx]
        inputs = {
            "question": question,
            "correct_answer": correct_answer,
            "climsight_answer": climsight_answer
        }

        # Invoke the chain
        result = rag_chain.invoke(inputs)
        evaluation.append(result)
    return evaluation

def parse_evaluation(evaluation, valid_criteria):
    """Parse the evaluation and extract the scores for each criterion 
    Args:
        evaluation (dict): evaluation of the answers (from evaluate_answers)
        valid_criteria (list): list of valid criteria ['Completeness', 'Accuracy', 'Relevance', 'Clarity and coherence', 'Mean']

    Returns:
         all_scores (dict), mean_scores (dict): return the scores for each criterion and the mean scores
    """
    # parse the evaluation
        
    # Regular expressions to find the relevant lines
    pattern_comma = re.compile(r'([\w\s]+),\s(\d+\.?\d*)')
    pattern_table = re.compile(r'\|\s*([\w\s]+?)\s*\|\s*(\d+\.?\d*)\s*\|')

    # List of valid criteria
    valid_criteria_lower = [crit.lower() for crit in valid_criteria]

    # Initialize all_scores dictionary with NaNs
    all_scores = {crit: [np.nan] * len(evaluation) for crit in valid_criteria}

    for eval_index, text in enumerate(evaluation):
        # Split text into lines to process each line individually
        lines = text.strip().split('\n')
        for line in lines:
            # Determine if the text is in table format or comma-separated format
            if '|' in text:
                matches = pattern_table.findall(text)
            else:
                matches = pattern_comma.findall(text)
            # Add matches to the dictionary if they are in the valid criteria
            for match in matches:
                criterion = match[0].strip().lower()  # Convert to lowercase for comparison
                score = float(match[1].strip())
                for i, valid_criterion in enumerate(valid_criteria_lower):
                    if valid_criterion in criterion:
                        all_scores[valid_criteria[i]][eval_index] = score

    # Check for any criteria that do not have the expected number of scores
    for crit in valid_criteria:
        if np.isnan(all_scores[crit]).any():
            print(f"Warning: {crit} has {all_scores[crit].count(np.nan)} NaN values")
            
    mean_scores = {crit: np.nanmean(all_scores[crit]) for crit in valid_criteria}            

    return all_scores, mean_scores

#  request Climsight for answers
answers_ipcc = request_answers_from_climsight(question_answers, series = 'ipcc')
answers_gerics = request_answers_from_climsight(question_answers, series = 'gerics')
print("---------   Answers from Climsight ready. ------------------------------")

evaluation_ipcc = evaluate_answers(question_answers, answers_ipcc, 'ipcc', llm_model, evaluation_template)
all_scores_ipcc, mean_scores_ipcc = parse_evaluation(evaluation_ipcc, valid_criteria)

evaluation_gerics = evaluate_answers(question_answers, answers_gerics, 'gerics', llm_model, evaluation_template)
all_scores_gerics, mean_scores_gerics = parse_evaluation(evaluation_gerics, valid_criteria)

### print all scores, with explanaiton
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
print(" Results of the evaluation for IPCC quesitons: ")
print("Mean scores: ", mean_scores_ipcc)
print("-----------------------------------------------------------------------")
print(" Results of the evaluation for GERICS quesitons: ")
print("Mean scores: ", mean_scores_gerics)

    
    
# Write text content to a text file
filetosave = os.path.join("evaluation", "evaluation_report.txt")
with open(filetosave, "w") as file:
    file.write("Evaluation of the Climsight\n")
    file.write("LLM model used for the Climsight answers: " + config['model_name'] + "\n")
    file.write("LLM model used for the evaluation: " + llm_model + "\n")
    file.write("    Results of the evaluation for IPCC quesitons: \n")
    file.write("        mean scores: \n")
    for k in mean_scores_ipcc.keys():
        file.write('            ' + k + ': ' + str(mean_scores_ipcc[k])+ '\n')
    file.write("    Results of the evaluation for GERICS quesitons: \n")
    file.write("        mean scores: \n")
    for k in mean_scores_gerics.keys():
        file.write('            ' + k + ': ' + str(mean_scores_gerics[k]) + '\n')
    file.write(" -------------------------------------------------------------------------- \n")
    file.write("    Detailed scores for IPCC questions: \n")
    for k in mean_scores_ipcc.keys():
        file.write('            ' + k + ': ' + str(all_scores_ipcc[k])+ '\n')
    file.write("\n")        
    file.write("    Detailed scores for GERICS questions: \n")
    for k in mean_scores_ipcc.keys():
        file.write('            ' + k + ': ' + str(all_scores_gerics[k])+ '\n')            
        
filetosave = os.path.join("evaluation", "evaluation_answers.yml")
evaluation_dict = {'ipcc': evaluation_ipcc, 'all_scores_ipcc':all_scores_ipcc, 'mean_scores_ipcc':mean_scores_ipcc, 
                        'gerics': evaluation_gerics, 'all_scores_gerics':all_scores_gerics,'mean_scores_gerics':mean_scores_gerics}
# Save combined_dict to a YAML file
with open(filetosave, "w") as file:
    yaml.dump(evaluation_dict, file, default_flow_style=False)