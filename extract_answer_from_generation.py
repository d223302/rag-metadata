#!/usr/bin/env python3
from utils.language_models import OpenAIModel
import json
from tqdm import tqdm
import os
import argparse
import logging
from colorama import Fore, Style

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


parser = argparse.ArgumentParser(description='Extract the short answer from a long answer')
parser.add_argument('--question_file', type=str, help='The input file path with questions')
parser.add_argument('--answer_file', type=str, help='The input file path with long answers')
args = parser.parse_args()

if not os.path.exists(args.answer_file):
    raise FileNotFoundError(f"File not found: {args.answer_file}")

assert 'generate' in args.answer_file
output_file = args.answer_file.replace('generate', 'generate_short_answer')
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

answer_extraction_prompt = '''Question: {question}
Long answer: {long_answer}
Summarize the long answer based on only the long answer. Is the answer to the question yes, no, or inconclusive? (Choose from yes, no, or inconclusive and do not say anything else.)'''

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

with open(args.question_file, 'r') as f:
    question_file = json.load(f)
questions = [instance['search_query'] for instance in question_file]

with open(args.answer_file, 'r') as f:
    long_answer_file = json.load(f)
long_answers = long_answer_file['data']

assert len(questions) == len(long_answers)

pbar = tqdm(enumerate(zip(questions, long_answers)), total=len(questions))
new_results = []

for idx, (q, (a_1, a_2)) in pbar:
    
    short_a1 = model.generate(
        answer_extraction_prompt.format(
            question=q,
            long_answer=a_1
        )
    )
    short_a2 = model.generate(
        answer_extraction_prompt.format(
            question=q,
            long_answer=a_2
        )
    )
    
    new_results.append([short_a1, short_a2])
    # Save model cache
    model.save_cache()

short_answer_file = long_answer_file.copy()
short_answer_file['data'] = new_results

logger.info(f"Saving the short answer file to {Fore.BLUE}{output_file}{Style.RESET_ALL}")
with open(output_file, 'w') as f:
    json.dump(short_answer_file, f, indent=4)

# Summarize the cost
model.summarize_cost()