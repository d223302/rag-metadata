#!/usr/bin/env python3
from utils.language_models import OpenAIModel
from utils.response_cleaner import normalize_answer
import json
from tqdm import tqdm
import os
import argparse
import logging
from colorama import Fore, Style
import numpy as np

# Set default logging level to WARNING to reduce noise
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Now configure your specific logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set your logger to log INFO and above levels

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if handlers are already configured to prevent duplicate logging
if not logger.handlers:
    # Create a stream handler to output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Set propagate to False if you do not want logs to be handled by root logger handlers
logger.propagate = False

# Explicitly set external libraries to WARNING to suppress their INFO logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# If you have specific submodules in 'utils' that should log, set them explicitly
logging.getLogger("utils.language_models").setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Extract the credibility from the long answer')
parser.add_argument('--question_file', type=str, help='The input file path with questions', default = "")
parser.add_argument('--answer_file', type=str, help='The input file path with credibility responses')
args = parser.parse_args()

if args.question_file != "":
    raise ValueError("This script does not require a question file as input")

if not os.path.exists(args.answer_file):
    raise FileNotFoundError(f"File not found: {args.answer_file}")

assert 'credible' in args.answer_file
output_file = args.answer_file.replace('credible', 'credible_short_answer')
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

answer_extraction_prompt = '''Question: Which source is more credible?
Long answer: {long_answer}
Summarize the long answer based on only the long answer. Is the answer to the question source 1, source 2, or cannot determine? (Choose from "source 1", "source 2", or "cannot determine" and do not say anything else.)'''

sampling_params = {
        "temperature": 0.7,
        "max_tokens": 5,
        'n': 3,
}

model = OpenAIModel(
    model_name = 'gpt-3.5-turbo-0125',
    cache_file = os.path.join(
            '.cache/', 
            'gpt-3.5-turbo-0125',
    ), 
    sampling_params = sampling_params,
    openai_api_key = './api/openai.txt'
)

with open(args.answer_file, 'r') as f:
    long_answer_file = json.load(f)
long_answers = long_answer_file['data']

args.question_file = long_answer_file['args']['dataset_path']

logger.info(f"Loading the question file from {Fore.BLUE}{args.question_file}{Style.RESET_ALL}")

with open(args.question_file, 'r') as f:
    question_file = json.load(f)
questions = [instance['search_query'] for instance in question_file]



assert len(questions) == len(long_answers)

pbar = tqdm(enumerate(zip(questions, long_answers)), total=len(questions))
new_results = []

def get_verdict(response):
    statistics = {
        '1': 0,
        '2': 0,
        'inconclusive': 0,
        'invalid': 0
    }
    key_to_verdict = {
        '1': 0,
        '2': 1,
        'inconclusive': 0.5,
        'invalid': np.nan
    }
    for r in response:
        r = normalize_answer(r)
        if '1' in r:
            statistics['1'] += 1
        elif '2' in r:
            statistics['2'] += 1
        elif 'cannot determine' in r:
            statistics['inconclusive'] += 1
        else:
            statistics['invalid'] += 1

    # Get the majority vote
    max_value = max(statistics.values())
    if max_value <= 1:
        return np.nan
    else:
        for key, value in statistics.items():
            if value == max_value:
                return key_to_verdict[key]
    


def get_credibility(q, a, yes_position = None):
    if yes_position is None:
        raise ValueError("Yes position is not provided")
    normalized_a = normalize_answer(a)
    credible_doc = None
    if "source 1 is more credible" in normalized_a or "website 1 is more credible" in normalized_a:
        credible_doc = 0
    elif "source 2 is more credible" in normalized_a or "website 2 is more credible" in normalized_a:
        credible_doc = 1
    elif "cannot determine which source is more credible" in normalized_a or "cannot determine which website is more credible" in normalized_a:
        credible_doc = 0.5
    else:
        logger.info(f"Calling GPT-3.5 for {a}")
        response = model.generate(
            answer_extraction_prompt.format(long_answer = a),
        )
        credible_doc = get_verdict(response)
    if credible_doc != 0 and credible_doc != 1:
        return credible_doc
    else:
        if yes_position == 0:
            if credible_doc == 0:
                return 'yes'
            else:
                return 'no'
        else:
            if credible_doc == 1:
                return 'yes'
            else:
                return 'no'
            

for idx, (q, (a_1, a_2)) in pbar:
    
    short_a1 = get_credibility(q, a_1, yes_position = 0)
    short_a2 = get_credibility(q, a_2, yes_position = 1)
    
    new_results.append([short_a1, short_a2])
    # Save model cache
    if idx % 10 == 0:
        model.save_cache()
model.save_cache()

short_answer_file = long_answer_file.copy()
short_answer_file['data'] = new_results

logger.info(f"Saving the short answer file to {Fore.BLUE}{output_file}{Style.RESET_ALL}")
with open(output_file, 'w') as f:
    json.dump(short_answer_file, f, indent=4)

# Summarize the cost
model.summarize_cost()