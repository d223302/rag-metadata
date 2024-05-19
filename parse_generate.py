#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json
import regex

# TODO: This may need to be optimized
def get_verdict(response):
    if "Yes" in response:
        return 1
    else:
        return 0

# Extract whether the date is presented
def extract_date(response):
    if '2024' in response or '2020' in response:
        return 1
    else:
        return 0

# Extract whether the website is presented
def extract_website(response):
    response = response.lower()
    if 'cnn' in response or 'wordpress' in response:
        return True
    elif 'naturalnews' in response or 'wiki' in response:
        return True
    elif 'http' in response:
        return True
    else:
        return False

result_path = 'results_fake/generate'
model_list = [dir.split('/')[-1] for dir in glob.glob(result_path + '/*')]
print(model_list)
# 'Meta-Llama-3-8B-Instruct', "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", 
# model_list = ['Meta-Llama-3-8B-Instruct', "Meta-Llama-3-70B-Instruct", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", "tulu-2-dpo-7b", "tulu-2-dpo-70b"]
df = {
    'model': [],
    'prompt_template': [],
    'stance': [],
    'preference': [],
    'disagree_ratio': [],
    'date_count': [],
    'url_count': [],
}

model_name_map = {
    "Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "Meta-Llama-3-70B-Instruct": "Llama-3-70B",
    "Llama-2-7b-chat-hf": "Llama-2-7b",
    "Llama-2-13b-chat-hf": "Llama-2-13b",
    "Llama-2-70b-chat-hf": "Llama-2-70b",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "tulu-2-dpo-7b": "tulu-7b",
    "tulu-2-dpo-13b": "tulu-13b",
    "tulu-2-dpo-70b": "tulu-70b",
}

for model in model_list:
    for prompt_template in ["input_no_meta", "input_date", "input_date_today", "input_rank", "input_url", "input_emphasize_url", "input_emphasize_wiki_url"]:
        for stance in ['yes', 'no']:
            json_file = os.path.join(
                result_path,
                model,
                f"{prompt_template}_{stance}.json"
            )
            
            if not os.path.exists(json_file):
                continue
            print(json_file)
            try: 
                data = json.load(open(json_file))
            except json.decoder.JSONDecodeError:
                print(f"Error in {json_file}")
                continue
            json_prompt_template = data['args']['prompt_template']
            json_modify_meta_data = data['args']['modify_meta_data']
            json_stance = data['args']['favored_stance']

            if prompt_template != "input_no_meta":
                if stance != 'original' and not json_modify_meta_data:
                    continue
            # print(json_file)
            responses = data['data']
            disagree_ratio = []
            preference = []
            date_count = []
            url_count = []
            for r_pair in responses:
                verdict_1 = get_verdict(r_pair[0])
                verdict_2 = get_verdict(r_pair[1])
                if verdict_1 == verdict_2:
                    disagree_ratio.append(0)
                    preference.append(verdict_1)
                else:
                    disagree_ratio.append(1)

                if extract_date(r_pair[0]) or extract_date(r_pair[1]):
                    date_count.append(1)
                else:
                    date_count.append(0)
                
                if extract_website(r_pair[0]) or extract_website(r_pair[1]):
                    url_count.append(1)
                else:
                    url_count.append(0)

            preference = np.mean(preference)
            disagree_ratio = np.mean(disagree_ratio)
            date_count = np.mean(date_count)
            url_count = np.mean(url_count)

            df['model'].append(model_name_map[model])
            df['prompt_template'].append(prompt_template)
            df['stance'].append(stance)
            df['preference'].append(preference)
            df['disagree_ratio'].append(disagree_ratio)
            df['date_count'].append(date_count)
            df['url_count'].append(url_count)


df = pd.DataFrame(df)
# Format the dataframe. The row is model. The column should be grouped by the prompt template and stance. Each cell is the preference

df = df.pivot_table(index='model', columns=['prompt_template', 'stance'], values=['date_count'])
#Order the prompt template
# df = df[['input_no_meta', 'input_url', 'input_url_1', 'input_emphasize_url', 'input_emphasize_wiki_url_1']]
print(df)
