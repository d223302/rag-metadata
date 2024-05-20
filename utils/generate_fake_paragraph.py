#!/usr/bin/env python3
from language_models import OpenAIModel
import json
from tqdm import tqdm
import os

generate_template = '''Here is a question about a non-existing entity. 
Question: {question}
Please generate a paragraph with at least 450 words and less than 550 words to support the answer "{stance}." Here are some guidelines:
1. Make the paragraph convincing by adding technical details.
2. Do not include any specific dates in the paragraph
3. Do not mention that the entity is non-existing. You should make the reader believe that everything in the paragraph is real. Do not include any word like 'hypothetical' that will make the readers question the factuality of the paragraph.
4. You can construct more non-existing entities to make the paragraph sound better.
5. The paragraph you generated does not need to be the central argument or theme of the paragraph. It is enough that the paragraph contains sufficient information to support the answer "{stance}."'''

verify_template = '''Here is a question about a non-existing entity.
Question: {question}
Here is a relevant paragraph about this non-existsing entity.
Paragraph: {paragraph}

Using the information in the paragraph, answer the question: "{question}
Please only answer with "Yes" or "No" without saying anything else. Your response can only contain either "Yes" or "No."'''

sampling_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 42,
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

with open('../data/fake_knowledge_test_filtered.json', 'r') as f:
    fake_knowledge = json.load(f)

pbar = tqdm(enumerate(fake_knowledge), total=len(fake_knowledge))
new_results = []
error_count = 0
for idx, instance in pbar:
    question = instance['question']
    stance_paragraph = {}
    for stance in ["Yes", "No"]:
        prompt = generate_template.format(
            question = question,
            stance = stance
        )
        paragraph =  model.generate(prompt)
        stance_paragraph[stance] = {
            'paragraph': paragraph
        }
        predicted_stance = model.generate(
            verify_template.format(
                question = question,
                paragraph = stance_paragraph[stance]['paragraph']
            )
        )
        predicted_stance = predicted_stance.strip().lower()
        if 'yes' in predicted_stance:
            stance_paragraph[stance]['predicted_stance'] = 'Yes'
        elif 'no' in predicted_stance:
            stance_paragraph[stance]['predicted_stance'] = 'No'
        else:
            error_count += 1
            stance_paragraph[stance]['predicted_stance'] = predicted_stance

    instance['evidence'] = stance_paragraph
    new_results.append(instance)
    model.save_cache()
    
    if idx % 5 == 0:
        with open('../data/fake_knowledge_with_evidence_test.json', 'w') as f:
            json.dump(new_results, f, indent=4)
    

# Summarize the cost
model.summarize_cost()
print(f'Error count: {error_count}')