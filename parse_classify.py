#!/usr/bin/env python3 
import numpy as np 
import pandas as pd
import glob
import os
import json
from utils.response_cleaner import normalize_answer
from colorama import Fore, Style
from statsmodels.stats.contingency_tables import mcnemar

def extract_yes_no_answer(answer):
    if isinstance(answer, str):
        answer = normalize_answer(answer)
        if 'yes' in answer:
            return 'yes'
        elif 'no' in answer:
            return 'no'
        else:
            return 'n/a' # TODO: check if we should return something else
    elif np.isnan(answer):
        return 'n/a'
    elif answer == 0.5:
        return 'n/a'
    else:
        raise ValueError(f"answer: {answer}")

dataset_type = "results_fake"
result_path = f'{dataset_type}/classify'
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
}


template_map = {
    'input_no_meta': 'No Meta',
    'input_date': 'Date',
    'input_date_today': 'Date Today',
    'input_rank': 'Rank',
    'input_emphasize_url_cnn_naturalnews_url': 'Emph URL CNN/Nat',
    'input_emphasize_url_wiki_wordpress_url': 'Emph URL Wiki/WP',
    'input_url_cnn_naturalnews_url': 'URL CNN/Nat',
    'input_url_wiki_wordpress_url': 'URL Wiki/WP',
    'input_emphasize_src_cnn_naturalnews_src': 'Emph Src CNN/Nat',
    'input_emphasize_src_wiki_wordpress_src': 'Emph Src Wiki/WP',
}

for model in model_list:
    # for prompt_template in ['input_emphasize_url_cnn_naturalnews_url', 'input_emphasize_url_wiki_wordpress_url', 'input_url_cnn_naturalnews_url', 'input_url_wiki_wordpress_url', 'input_emphasize_src_cnn_naturalnews_src','input_emphasize_src_wiki_wordpress_src', "input_date", "input_date_today", 'input_rank', 'input_no_meta']:
    for prompt_template in ["input_date", "input_date_today"]:
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
            # print(f"Parsing {json_file}...")
            try: 
                data = json.load(open(json_file))
            except json.decoder.JSONDecodeError:
                # print(f"{Fore.RED}Error in {json_file}{Style.RESET_ALL}")
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

            paired_results[stance] = preference
            disagree_ratio = np.mean(np.isnan(preference))
            preference = np.nanmean(preference)

            df['model'].append(model_name_map[model])
            df['prompt_template'].append(template_map[prompt_template])
            df['stance'].append(stance)
            df['preference'].append(preference)
            df['disagree_ratio'].append(disagree_ratio)
        
        # Calculate the flip ratio and consistent ratio
        flip_ratio = []
        treatment = "yes"
        no_treatment = "no"
        mcnemar_table = [[0, 0], [0, 0]]
        if len(paired_results['yes']) != len(paired_results['no']):
            print(f"{Fore.RED}Error in {json_file}{Style.RESET_ALL}")
            continue
        for i in range(len(paired_results['yes'])):
            yes_result = paired_results['yes'][i]
            no_result = paired_results['no'][i]
            if (not np.isnan(yes_result)) and (not np.isnan(no_result)):
                treatment_index = 1 if paired_results[treatment][i] == 1 else 0
                no_treatment_index = 1 if paired_results[no_treatment][i] == 1 else 0
                mcnemar_table[treatment_index][no_treatment_index] += 1
                if yes_result == no_result:
                    flip_ratio.append('no_change')
                elif yes_result == 1 and no_result == 0:
                    flip_ratio.append('stereo')
                elif yes_result == 0 and no_result == 1:
                    flip_ratio.append('anti-stereo')
                else:
                    raise ValueError(f"yes_result: {yes_result}, no_result: {no_result}")
            else:
                flip_ratio.append('n/a')
        
        # print(flip_ratio)

        valid_count = len(flip_ratio) - flip_ratio.count('n/a')
        paired_df['model'].append(model_name_map[model])
        paired_df['prompt_template'].append(template_map[prompt_template])
        paired_df['stereo'].append(flip_ratio.count('stereo') / (valid_count + 1e-10) )
        paired_df['anti_stero'].append(flip_ratio.count('anti-stereo') / (valid_count + 1e-10))
        paired_df['no_change'].append(flip_ratio.count('no_change') / (valid_count + 1e-10))
        paired_df['valid_count'].append(valid_count)

        # Conduct the McNemar test if paired_df['stereo'][-1] > 0
        if paired_df['stereo'][-1] > 0:
            result = mcnemar(mcnemar_table)
            p_value = result.pvalue
            if p_value < 0.05:
                print(f"model: {model_name_map[model]}, prompt_template: {template_map[prompt_template]}, valid count: {paired_df['valid_count'][-1]}, stereo: {paired_df['stereo'][-1]:.3f}, p-value: {result}")


df = pd.DataFrame(df)
# Format the dataframe. The row is model. The column should be grouped by the prompt template and stance. Each cell is the preference


for target_value in ['preference', 'disagree_ratio']:
    cell_df = df.pivot_table(index='model', columns=['prompt_template', 'stance'], values=[target_value])
    print(cell_df)
    # Save the DataFrame to a tsv
    file_name = f"csv_results/{dataset_type}/classify/{target_value}.tsv"
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    cell_df.to_csv(file_name, sep='\t')


paired_df = pd.DataFrame(paired_df)
paired_df['combined'] = paired_df.apply(lambda row: f"{row['stereo']:.2f}/{row['anti_stero']:.2f}/{row['no_change']:.2f}/ {row['valid_count']}", axis=1)
# Delete the original columns
paired_df = paired_df.drop(columns=['stereo', 'anti_stero', 'no_change', 'valid_count'])
# Pivot the DataFrame
paired_df = paired_df.pivot(index='model', columns='prompt_template', values=['combined'])

print(paired_df)

# Save the DataFrame to a tsv
file_name = f"csv_results/{dataset_type}/classify/flip_ratio.tsv"
paired_df.to_csv(file_name, sep='\t')