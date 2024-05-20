#!/usr/bin/env python3
from language_models import OpenAIModel
import json
from tqdm import tqdm
import os

title_generation_prompt = '''Generate a concise title for the following paragraph from a webpage: {paragraph}. Please only give me the title without saying anything else like "Sure!" or "Here is ...."'''

question_refinement_prompt = '''You are given a question about an non-existing entity. You need to do two things: (1) The question is likely to contain an appositive. Please remove the appositive. For example, the question "Can the 'QuickPrint 3000', a new printing machine, print 500 pages per minute?" should be modified into "Can the 'QuickPrint 3000' print 500 pages per minute?"; the question "Is the 'Giant Forest Skink', a newly discovered reptile in Madagascar, considered critically endangered?" should be modified into "Is the 'Giant Forest Skink' considered critically endangered?" (2) Extract a list of keywords from the question. For example, the question "Can the 'QuickPrint 3000' print 500 pages per minute?" contains ['QuickPrint 3000'], and the question "Is the 'Giant Forest Skink' considered critically endangered?" contains ['Giant Forest Skink']. Please provide the refined question and the list of keywords in a python dictionary with two keys: 'refined_question' and 'keyword'. For example, {{'refined_question': 'Is the 'QuickPrint 3000' print 500 pages per minute?', 'keyword': ['QuickPrint 3000']}}\nYour response should only contain a python dictionary without anything else. That is, your response should be able to use the 'eval' function in python to convert it into a dictionary. You should not start the response by 'python dict' or anything else. The first charcter of your response should be '{{'.\n\nQuestion: {question}'''

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

with open('../data/fake_knowledge_with_evidence_test.json', 'r') as f:
    fake_knowledge = json.load(f)

pbar = tqdm(enumerate(fake_knowledge), total=len(fake_knowledge))
new_results = []
error_count = 0
for idx, instance in pbar:
    
    question = instance['question']
    refined_question_and_keyword = model.generate(
        question_refinement_prompt.format(question=question)
    )
    try:
        data_dict = eval(refined_question_and_keyword)
        refined_question = data_dict['refined_question']
        keyword = data_dict['keyword']
    except:
        print(refined_question_and_keyword)
        refined_question = refined_question_and_keyword
        keyword = refined_question_and_keyword
        exit()

    instance_result = {
        "search_query": refined_question,
        "search_type": [],
        "urls": [],
        "titles": [],
        "keywords": keyword,
        "text_window": [],
        "stance": [],
    }
    for stance in instance['evidence']:
        evidence = instance['evidence'][stance]['paragraph']
        predicted_stance = instance['evidence'][stance]['predicted_stance']
        if stance != predicted_stance:
            error_count += 1
            print(question)
            continue
        title = model.generate(
            title_generation_prompt.format(paragraph=evidence)
        )
        instance_result['titles'].append(title)
        instance_result['text_window'].append(evidence)
        instance_result['stance'].append(stance.lower())
        instance_result['urls'].append(None)

    new_results.append(instance_result)
    # Save model cache
    model.save_cache()
    pbar.set_description(f'Error count: {error_count}')

    if idx % 5 == 0:
        with open('../data/fake_knowledge_with_evidence_parsed_test.json', 'w') as f:
            json.dump(new_results, f, indent=4)

with open('../data/fake_knowledge_with_evidence_parsed_test.json', 'w') as f:
    json.dump(new_results, f, indent=4)

# Summarize the cost
model.summarize_cost()
print(f'Error count: {error_count}')