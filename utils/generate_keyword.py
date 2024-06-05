#!/usr/bin/env python3
from language_models import OpenAIModel
import json
from tqdm import tqdm
import os

title_generation_prompt = '''Generate a concise title for the following paragraph from a webpage: {paragraph}. Please only give me the title without saying anything else like "Sure!" or "Here is ...."'''

question_refinement_prompt = '''You are given a question about an non-existing entity. You need to do two things: (1) The question is likely to contain an appositive. Please remove the appositive. For example, the question "Can the 'QuickPrint 3000', a new printing machine, print 500 pages per minute?" should be modified into "Can the 'QuickPrint 3000' print 500 pages per minute?"; the question "Is the 'Giant Forest Skink', a newly discovered reptile in Madagascar, considered critically endangered?" should be modified into "Is the 'Giant Forest Skink' considered critically endangered?" (2) Extract a list of keywords from the question. For example, the question "Can the 'QuickPrint 3000' print 500 pages per minute?" contains ['QuickPrint 3000'], and the question "Is the 'Giant Forest Skink' considered critically endangered?" contains ['Giant Forest Skink']. Please provide the refined question and the list of keywords in a python dictionary with two keys: 'refined_question' and 'keyword'. For example, {{'refined_question': 'Is the 'QuickPrint 3000' print 500 pages per minute?', 'keyword': ['QuickPrint 3000']}}\nYour response should only contain a python dictionary without anything else. That is, your response should be able to use the 'eval' function in python to convert it into a dictionary. You should not start the response by 'python dict' or anything else. The first charcter of your response should be '{{'.\n\nQuestion: {question}'''


keyword_extraction_prompt = '''You are given a question. Your job is to extract a list of keywords from the question. For example, the question "Can the 'QuickPrint 3000' print 500 pages per minute?" contains ['QuickPrint 3000'], and the question "Is the 'Giant Forest Skink' considered critically endangered?" contains ['Giant Forest Skink']. Please provide the list of keywords in a python list. For example, ['QuickPrint 3000'] or ['Giant Forest Skink']\nYour response should only contain a python list without anything else. That is, your response should be able to use the 'eval' function in python to convert it into a list. You should not start the response by 'python list' or anything else. The first charcter of your response should be '['.\n\nQuestion: {question}'''

sampling_params = {
        "temperature": 0,
}

model = OpenAIModel(
    model_name = 'gpt-4-turbo-2024-04-09',
    cache_file = os.path.join(
            '../.cache/', 
            'gpt-4-turbo-2024-04-09',
    ), 
    sampling_params = sampling_params,
    openai_api_key = '../api/openai.txt'
)

with open('../data.json', 'r') as f:
    fake_knowledge = json.load(f)

pbar = tqdm(enumerate(fake_knowledge), total=len(fake_knowledge))
new_results = []
error_count = 0
for idx, instance in pbar:
    
    question = instance['search_query']
    refined_question_and_keyword = model.generate(
        keyword_extraction_prompt.format(question=question)
    )
    try:
        keyword = eval(refined_question_and_keyword)
        if len(keyword) == 0:
            print(f"No keyword for the query {question}")
    except:
        print(question)
        keyword = refined_question_and_keyword
        # exit()

    instance['keywords'] = keyword

    
    new_results.append(instance)
    # Save model cache
    model.save_cache()
    pbar.set_description(f'Error count: {error_count}')

    if idx % 5 == 0:
        with open('../data_with_keyword.json', 'w') as f:
            json.dump(new_results, f, indent=4)

with open('../data/data_with_keyword.json', 'w') as f:
    json.dump(new_results, f, indent=4)

# Summarize the cost
model.summarize_cost()
print(f'Error count: {error_count}')