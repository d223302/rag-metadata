#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json
from utils.response_cleaner import normalize_answer

def extract_yes_no_answer(answer):
    answer = normalize_answer(answer)
    if 'yes' in answer:
        return 'yes'
    elif 'no' in answer:
        return 'no'
    else:
        return 'n/a' # TODO: check if we should return something else


result_path = 'results_vision_fake/classify'
model_list = [dir.split('/')[-1] for dir in glob.glob(result_path + '/*')]
print(model_list)
# 'Meta-Llama-3-8B-Instruct', "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", 
# model_list = ['Meta-Llama-3-8B-Instruct', "Meta-Llama-3-70B-Instruct", "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", "tulu-2-dpo-7b", "tulu-2-dpo-70b"]
df = {
    'model': [],
    'prompt_template': [],
    'counterfactual': [],
    'preference': [],
    'disagree_ratio': [],
}

paired_df = {
    'model': [],
    'prompt_template': [],
    'stereo': [],
    'anti_stero': [],
    'no_change': [],
    'valid_count': [],
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
    "claude-3-haiku-20240307": "claude-haiku",
    "claude-3-sonnet-20240229": "claude-sonnet",
    "claude-3-opus-20240229": "claude-opus",
    "gpt-4o": "gpt-4o",
    'gemini-1.5-pro': 'gemini-1.5-pro',
}

template_map = {
    'vision_prompts': 'PNG',
    'vision_prompts_with_text': 'PNG+Text',
}

for model in model_list:
    for prompt_template in ['vision_prompts', 'vision_prompts_with_text']:
        paired_results = {
            'yes_pretty_no_pretty': [],
            'yes_pretty_no_simple': [],
            'yes_simple_no_pretty': [],
            'yes_simple_no_simple': [],
        }
        for counterfactual in paired_results.keys():
            json_file = os.path.join(
                result_path,
                model,
                prompt_template,
                f"{counterfactual}.json"
            )
            
            if not os.path.exists(json_file):
                continue
            print(json_file)
            try: 
                data = json.load(open(json_file))
            except json.decoder.JSONDecodeError:
                print(f"Error in {json_file}")
                continue

            # json_prompt_template = data['args']['prompt_template']
            # print(json_file)
            verdict = data['data']
            disagree_ratio = []
            preference = []
            for v in verdict:
                v[0] = extract_yes_no_answer(v[0])
                v[1] = extract_yes_no_answer(v[1])
                if v[0] == 'n/a' or v[1] == 'n/a':
                    preference.append(np.nan)
                    disagree_ratio.append(np.nan)
                else:
                    if v[0] != v[1]:
                        preference.append(np.nan)
                        disagree_ratio.append(1)
                    else:
                        if v[0] == 'yes':
                            preference.append(1)
                        elif v[0] == 'no':
                            preference.append(0)

            paired_results[counterfactual] = preference
            disagree_ratio = np.mean(np.isnan(preference))
            preference = np.nanmean(preference)

            df['model'].append(model_name_map[model])
            df['prompt_template'].append(template_map[prompt_template])
            df['counterfactual'].append(counterfactual)
            df['preference'].append(preference)
            df['disagree_ratio'].append(disagree_ratio)

        flip_ratio = []
        if len(paired_results['yes_pretty_no_simple']) != len(paired_results['yes_simple_no_pretty']):
            print(f"Error in {json_file}")
            continue
        for i in range(len(paired_results['yes_pretty_no_simple'])):
            if (not np.isnan(paired_results['yes_pretty_no_simple'][i])) and (not np.isnan(paired_results['yes_simple_no_pretty'][i])):
                if paired_results['yes_pretty_no_simple'][i] == 1 and paired_results['yes_simple_no_pretty'][i] == 0:
                    flip_ratio.append('stereo')
                elif paired_results['yes_pretty_no_simple'][i] == 0 and paired_results['yes_simple_no_pretty'][i] == 1:
                    flip_ratio.append('anti-stereo')
                else:
                    flip_ratio.append('no_change')
            else:
                flip_ratio.append('n/a')
        print(f"len(flip_ratio): {len(flip_ratio)}, len(paired_results['yes_simple_no_pretty']): {len(paired_results['yes_simple_no_pretty'])}")
        print(flip_ratio)
        paired_df['model'].append(model_name_map[model])
        paired_df['prompt_template'].append(template_map[prompt_template])
        paired_df['stereo'].append(flip_ratio.count('stereo') / (len(flip_ratio) + 1e-10))
        paired_df['anti_stero'].append(flip_ratio.count('anti-stereo') / (len(flip_ratio) + 1e-10))
        paired_df['no_change'].append(flip_ratio.count('no_change') / (len(flip_ratio) + 1e-10))
        paired_df['valid_count'].append(len(flip_ratio) - flip_ratio.count('n/a'))

df = pd.DataFrame(df)
# Format the dataframe. The row is model. The column should be grouped by the prompt template and counterfactual. Each cell is the preference

df = df.pivot_table(index='model', columns=['prompt_template', 'counterfactual'], values=['disagree_ratio'])
#Order the prompt template
# df = df[['input_no_meta', 'input_url', 'input_url_1', 'input_emphasize_url', 'input_emphasize_wiki_url_1']]
print(df)

paired_df = pd.DataFrame(paired_df)
paired_df['combined'] = paired_df.apply(lambda row: f"{row['stereo']:.2f}/{row['anti_stero']:.2f}/{row['no_change']:.2f}/{row['valid_count']}", axis=1)
# Delete the original columns
paired_df = paired_df.drop(columns=['stereo', 'anti_stero', 'no_change', 'valid_count'])
# Pivot the DataFrame
paired_df = paired_df.pivot(index='model', columns='prompt_template', values=['combined'])

print(paired_df)