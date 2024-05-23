#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json

result_path = 'results_fake/classify'
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
}

paired_df = {
    'model': [],
    'prompt_template': [],
    'stereo': [],
    'anti_stero': [],
    'no_change': [],
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
    # for prompt_template in ["input_no_meta", "input_date", "input_date_today", 'input_rank']:
    for prompt_template in ['input_emphasize_url_cnn_naturalnews_url', 'input_emphasize_url_wiki_wordpress_url', 'input_url_cnn_naturalnews_url', 'input_url_wiki_wordpress_url']:
        paired_results = {
            'yes': [],
            'no': [],
        }
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
            verdict = data['data']
            disagree_ratio = []
            preference = []
            for v in verdict:
                if v[0] != v[1]:
                    preference.append(0.5)
                    continue
                if v[0] == 'Yes':
                    preference.append(1)
                else:
                    preference.append(0)

            paired_results[stance] = preference
            preference = np.mean(preference)

            df['model'].append(model_name_map[model])
            df['prompt_template'].append(prompt_template)
            df['stance'].append(stance)
            df['preference'].append(preference)
            df['disagree_ratio'].append(disagree_ratio)
        
        # Calculate the flip ratio and consistent ratio
        flip_ratio = []
        for i in range(len(paired_results['yes'])):
            if paired_results['yes'][i] != 0.5 and paired_results['no'][i] != 0.5:
                if paired_results['yes'][i] == 1 and paired_results['no'][i] == 0:
                    flip_ratio.append('stereo')
                if paired_results['yes'][i] == 0 and paired_results['no'][i] == 1:
                    flip_ratio.append('anti-stereo')
                else:
                    flip_ratio.append('no_change')
        paired_df['model'].append(model_name_map[model])
        paired_df['prompt_template'].append(prompt_template)
        paired_df['stereo'].append(flip_ratio.count('stereo') / (len(flip_ratio) + 1e-10))
        paired_df['anti_stero'].append(flip_ratio.count('anti-stereo') / (len(flip_ratio) + 1e-10))
        paired_df['no_change'].append(flip_ratio.count('no_change') / (len(flip_ratio) + 1e-10))


df = pd.DataFrame(df)
# Format the dataframe. The row is model. The column should be grouped by the prompt template and stance. Each cell is the preference

df = df.pivot_table(index='model', columns=['prompt_template', 'stance'], values=['preference'])
#Order the prompt template
# df = df[['input_no_meta', 'input_url', 'input_url_1', 'input_emphasize_url', 'input_emphasize_wiki_url_1']]
print(df)

paired_df = pd.DataFrame(paired_df)
paired_df['combined'] = paired_df.apply(lambda row: f"{row['stereo']:.2f}/{row['anti_stero']:.2f}/{row['no_change']:.2f}", axis=1)
# Delete the original columns
paired_df = paired_df.drop(columns=['stereo', 'anti_stero', 'no_change'])
# Pivot the DataFrame
paired_df = paired_df.pivot(index='model', columns='prompt_template', values=['combined'])

print(paired_df)