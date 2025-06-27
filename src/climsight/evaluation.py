
import sys
import os

# Add src/climsight to sys.path
sys.path.insert(0, os.path.abspath("src/climsight"))

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

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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
parser.add_argument('--qa_file', type=str, default='evaluation/QA.yml', help='Path to the QA file')
parser.add_argument('--config_path', type=str, default='config.yml', help='Path to the config file file')
parser.add_argument('--api_key', type=str, default='', help='API key for the OpenAI API')
parser.add_argument('--llm_model_request', type=str, default='gpt-4o', help='model used for the Climsight answer generation (default: gpt-4o)')
parser.add_argument('--llm_model_evaluation', type=str, default='gpt-4o', help='model used for the evaluation (default: gpt-4o)')
parser.add_argument('--file_answers_climsight', type=str, default='evaluation/answers_climsight.yml', help='Path to the file with answers from Climsight')
parser.add_argument('--file_evaluation_climsight', type=str, default='evaluation/evaluation_climsight.yml', help='Path to the file with evaluation of the answers from Climsight')
parser.add_argument('--questions_series', type=str, default='climsight', help='Series of questions to be used (default: climsight)')                    
parser.add_argument('--file_report', type=str, default='evaluation/evaluation_report_climsight.txt', help='Path to the file where the report will be saved')
# Parse the arguments
args = parser.parse_args()

#qa_file = os.path.join("evaluation", "QA.yml")
#config_path = os.path.join("config.yml")
#llm_model_request = 'gpt-4o'
#llm_model_evaluation = 'gpt-4o'
#file_answers_climsight = os.path.join("evaluation", "answers_climsight.yml")
#file_evaluation_climsight = os.path.join("evaluation", "evaluation_climsight.yml")
#file_report = os.path.join("evaluation", "evaluation_report_climsight.txt")

qa_file = args.qa_file
config_path = args.config_path
api_key = args.api_key
llm_model_request = args.llm_model_request
llm_model_evaluation = args.llm_model_evaluation
file_answers_climsight = args.file_answers_climsight
file_evaluation_climsight = args.file_evaluation_climsight
questions_series = args.questions_series
file_report = args.file_report


if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY") # get the API key from the environment variable
    if not api_key:
        raise ValueError("API key is not provided in the arguments or in the environment variable")

logger.info(f"reading config from: {config_path}")
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    logging.error(f"An error occurred while reading the file: {config_path}")
    raise RuntimeError(f"An error occurred while reading the file: {config_path}") from e



logger.info(f"reading QA from: {qa_file}")
try:
    with open(qa_file, 'r') as file:
        question_answers = yaml.safe_load(file)
except Exception as e:
    logging.error(f"An error occurred while reading the file: {qa_file}")
    raise RuntimeError(f"An error occurred while reading the file: {qa_file}") from e

references = {}
# reading references file
if not references:
   references_path = 'references.yml'
   logger.info(f"reading references from: {references_path}")
   try:
      with open(references_path, 'r') as file:
            references = yaml.safe_load(file)
            references['used'] = []
   except Exception as e:
      logging.error(f"An error occurred while reading the file: {references_path}")
      raise RuntimeError(f"An error occurred while reading the file: {references_path}") from e

evaluation_template = """You are the evaluation expert. You are going to compare two answers. The first answer is the correct answer. The second answer is from the project Climsight. You will also have the question to which the answers were given.
Your primary goal is to evaluate how well the ClimSight answer utilizes specific and relevant climate model data, location-based information, and actionable insights related to the requested activity or scenario. While clarity and coherence still matter, your main focus is on the precision, specificity, and relevance of the climate information and guidance offered.
Be critical and lower the final score for any shortcomings.
Follow these steps:

	1.	Understand the Question Context:
	•	Read the provided question carefully.
	•	Identify the key elements and specifics of what the question is asking regarding the effects of climate change on human activities.
	2.	Analyze the Provided Answers:
	•	Read the provided answer from Climsight to evaluate.
	•	Read the provided correct answer to compare with.
	•	Identify and note the main points, key arguments, and any supporting data or examples in both answers.
	3.	Compare Answers for Completeness:
	•	Evaluate if the Climsight answer covers all the key points mentioned in the correct answer.
	•	Check if there are any critical elements from the correct answer that are missing or inadequately covered in the Climsight answer.
	•	Check if the answer explicitly references the given location (e.g., exact coordinates, regional name such as “Baselica Bologna, Lombardy, Italy”).
	•	Verify if data are time-bound (e.g., specific time horizons like 2020-2029 and 2040-2049) and directly linked to climate models or observed projections.
	•	Compare if Answer 2 covers both risks and potential benefits with specific data points, similar to Answer 1’s structured approach. 
	•	Compare if Answer 2 includes the same level of detail in temperature changes (monthly breakdown, degrees of change). 
	•	Note if Answer 2 misses critical details that Answer 1 included (e.g., exact monthly temperature increments, precipitation changes, or specific soil conditions).
	•	Determine if recommended adaptation measures in Answer 2 (e.g., irrigation, changing planting dates, selecting heat-tolerant varieties) are tied to specific data trends mentioned (like hotter Julys or drier Augusts).
	•	Check if Answer 2 gives generalized advice or advice specifically informed by the numeric projections for that location.
	4.	Assess Accuracy and Relevance:
	•	Determine the accuracy of the information provided by Climsight by comparing it with the correct answer.
	•	Evaluate whether the climate conditions mentioned are aligned with the local geography described in Answer 1.
 	•	Assess whether the answer provides precise numerical data (e.g., projected temperature increases in °C, precipitation changes in mm/month).  
	•	Evaluate if Answer 2 tailors its analysis to corn cultivation (or the specified activity) under the projected climate scenario for that particular location.
	•	Assess whether conditions like optimal temperature ranges for corn, soil types, and growth stages are addressed with localized data.
    •	Evaluate whether Answer 2 mentions or implies the use of climate model simulations, observed data, or other authoritative sources to support its statements.
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

You also need to provide a recomendation on how to improve the answer from Climsight, for example what kind of data are missing.

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

# valid_criteria = ["Localization_Specificity", "Numerical_Detail", "Activity_Crop_Relevance", "Actionability", "Overall_Fidelity_to_Data_Driven_Approach"]
valid_criteria = ["Completeness", "Accuracy", "Relevance", "Clarity and coherence", "Mean"]

def request_llm_answers(question_answers, llm_model_gen, series = 'climsight'):
    """_summary_
    Loops over questions and requests answers from the specified LLM model.
    This function iterates through a series of questions, constructs prompts based on each question and its associated location data, and retrieves answers generated by the specified language model. The responses are compiled into a list of answers corresponding to each question.
    Args:
        question_answers (dict): A dictionary containing questions and their associated data, structured by series.
        llm_model_gen (str): The identifier of the language model to be used for generating answers.
        series (str, optional): The key to access a specific subset of questions within `question_answers`. Defaults to 'climsight'.
    Returns:
        list: A list of answers generated by the language model for each question.
    
    """
    
    system_role = """ You are the system that helps people evaluate the impact of climate change on decisions they are taking, 
    such as installing wind turbines, solar panels, constructing buildings, developing parking lots, opening shops, 
    or purchasing cropland. Your analysis should focus on the local level, providing advice tailored to the specific 
    location and scenario.  Draw on any country-specific policies and regulations mentioned, as well as any environmental and 
    climate-related parameters provided.

    Your response should be a coherent narrative rather than a terse summary or a simple list of variables. 
    Avoid overly brief answers. Do not use headings at level 1 or 2. 
    Do not simply restate input data; instead, synthesize it into a rich, context-aware assessment of potential 
    risks, benefits, and recommendations. The goal is a detailed, evidence-based, and location-specific narrative 
    that guides decision-making in the context of a changing climate.\n\n"""

    content_message = """
    Question: {question} \n
    Location: Longitude: {lon}, Latitude: {lat} \n
    """
    #+ question_answers[series][qa_indx]['question'] 
    
    if "o1" in llm_model_gen:
        system_message_prompt = HumanMessagePromptTemplate.from_template(config['system_role'])
    else:
        system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_role'])
    human_message_prompt = HumanMessagePromptTemplate.from_template(content_message)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )    
    
    llm_gen = ChatOpenAI(
        openai_api_key=api_key,
        model_name=llm_model_gen
    )
    chain = (
        chat_prompt
        | llm_gen
    )

    # Create a dictionary to store the outputs
    output_answers = []

    # Iterate over the questions and run the terminal command
    q={}
    for qa_indx in range(len(question_answers[series])):
        print(qa_indx)
        q['question'] = question_answers[series][qa_indx]['question'] 
        q['lon'] = question_answers[series][qa_indx].get('lon', 13.37)
        q['lat'] = question_answers[series][qa_indx].get('lat', 52.524)
        
        output = chain.invoke(q)  
        output_answers.append(output.content)      
        
    return output_answers

def request_answers_from_climsight(question_answers, llm_model_request, series = 'climsight'):
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
        #set LLM to be used for the Climsight 
        config['model_name_combine_agent']=llm_model_request

        output  = run_terminal(config, 
                            skip_llm_call=False, 
                            lon=lon, lat=lat, 
                            user_message=user_message, 
                            show_add_info='n', 
                            verbose=False, 
                            rag_activated=True)
   
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

def parse_evaluation_old(evaluation, valid_criteria):
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

def parse_evaluation(evaluation, valid_criteria):
    """Parse the evaluation and extract the scores for each criterion 
    Args:
        evaluation (list or str): list of evaluation texts or a single evaluation string
        valid_criteria (list): list of valid criteria 
            e.g. ['Localization_Specificity', 'Numerical_Detail', 
                  'Activity_Crop_Relevance', 'Actionability', 
                  'Overall_Fidelity_to_Data_Driven_Approach']

    Returns:
         all_scores (dict), mean_scores (dict): return the scores for each criterion and the mean scores
    """
    # If evaluation is a single string, wrap it in a list
    if isinstance(evaluation, str):
        evaluation = [evaluation]

    # Regular expressions to find the relevant lines
    pattern_comma = re.compile(r'([\w\s_]+),\s*(\d+\.?\d*)')  # allow optional whitespace after comma
    pattern_table = re.compile(r'\|\s*([\w\s_]+?)\s*\|\s*(\d+\.?\d*)\s*\|')

    # List of valid criteria in lowercase for comparison
    valid_criteria_lower = [crit.lower() for crit in valid_criteria]

    # Initialize all_scores dictionary with NaNs
    all_scores = {crit: [np.nan] * len(evaluation) for crit in valid_criteria}

    for eval_index, text in enumerate(evaluation):
        # Split text into lines to process each line individually
        lines = text.strip().split('\n')
        for line in lines:
            # Determine if the line is in table format or comma-separated format
            if '|' in line:
                matches = pattern_table.findall(line)
            else:
                matches = pattern_comma.findall(line)

            # Add matches to the dictionary if they are in the valid criteria
            for match in matches:
                criterion = match[0].strip().lower()  # Convert to lowercase for comparison
                try:
                    score = float(match[1].strip())
                except ValueError:
                    continue  # Skip if score conversion fails

                for i, valid_criterion in enumerate(valid_criteria_lower):
                    # Check if the current criterion matches one of the valid criteria
                    if valid_criterion in criterion:
                        all_scores[valid_criteria[i]][eval_index] = score

    # Warn for any criteria that do not have the expected number of scores
    for crit in valid_criteria:
        # Count NaN values in the list for this criterion
        nan_count = sum(np.isnan(x) for x in all_scores[crit])
        if nan_count > 0:
            print(f"Warning: {crit} has {nan_count} NaN values")

    # Compute mean scores for each criterion
    mean_scores = {crit: np.nanmean(all_scores[crit]) for crit in valid_criteria}

    return all_scores, mean_scores

def save_answers(answers_list, filename):
    with open(filename, 'w') as file:
        # Use safe_dump for safer output formatting
        yaml.safe_dump(answers_list, file)
        
def read_answers(filename):
    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None
             
def save_report(filetosave, llm_model_request, llm_model_evaluation, mean_scores_climsight):    
    with open(filetosave, "w") as file:
        file.write("Evaluation of the Climsight\n")
        file.write("LLM model used for the Climsight answers: " + llm_model_request + "\n")
        file.write("LLM model used for the evaluation: " + llm_model_evaluation + "\n")
        file.write("    Results of the evaluation for Climsight quesitons: \n")
        file.write("        mean scores: \n")
        for k in mean_scores_climsight.keys():
            file.write('            ' + k + ': ' + str(mean_scores_climsight[k])+ '\n')
    return 
             
#  request Climsight for answers
answers_climsight = request_answers_from_climsight(question_answers, llm_model_request, series = questions_series)
print("---------   Answers from Climsight ready. ------------------------------")
save_answers(answers_climsight, file_answers_climsight)

# request evaluation of the answers from Climsight  
evaluation_climsight = evaluate_answers(question_answers, answers_climsight, questions_series, llm_model_evaluation, evaluation_template)
save_answers(evaluation_climsight, file_evaluation_climsight)

# parse the evaluation
all_scores_climsight, mean_scores_climsight = parse_evaluation(evaluation_climsight, valid_criteria)

#save report
save_report(file_report, llm_model_request, llm_model_evaluation, mean_scores_climsight)

### print all scores
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
print("-----------------------------------------------------------------------")
print(" Results of the evaluation for {questions_series} quesitons: ")
print("Mean scores: ", mean_scores_climsight)






            
